# Ask a Coast Guard Lawyer
### RAG System for USCG Law and Policy

**2/c Michael Porter · 2/c Christian Gottlieb**
Project Based Learning in AI · CGA Spring 2026

---

A Retrieval-Augmented Generation system that answers questions about U.S. Coast Guard
and military law. Retrieves from official source documents, generates plain-language
summaries grounded strictly in those sources, and cites every claim to a specific
document, section, and page number.

> **This system is not a substitute for legal counsel.**
> Every response includes an explicit disclaimer directing users to their legal office.

---

## Retrieval Performance

Evaluated on a 200-question benchmark spanning all six ingested documents.

| Pipeline | HIT@1 | MRR | Recall@8 | Precision@3 |
|---|---|---|---|---|
| Dense only (baseline) | 109 / 200 (54.5%) | 0.660 | 88.5% | 0.392 |
| Hybrid BM25 + Dense | 134 / 200 (67.0%) | 0.768 | 98.0% | 0.575 |
| **Hybrid + Cross-Encoder Rerank** | **142 / 200 (71.0%)** | **0.800** | **99.0%** | **0.593** |

- **HIT@1 = 71%** — the correct source is the top result in 7 out of 10 queries
- **Recall@8 = 99%** — the correct source appears in the top 8 results for 99% of queries
- **MRR = 0.800** — the correct result appears at an average rank of 1.25

Benchmark: `eval/benchmark_200.json` — 200 questions, 7 categories, 6 documents,
20+ gold label corrections applied by content inspection.

---

## Corpus

| Document | Citation | Chunks | Splitting boundary |
|---|---|---|---|
| MCM 2024 Punitive Articles | Part IV, Arts. 77–134 | 292 | Article number |
| CG Conduct Manual | COMDTINST M1600.2 | 158 | Section number `1.A.3.` |
| CG Administrative Investigations Manual | COMDTINST M5830.1A | 189 | Letter per chapter `A.` |
| CG Military Separations Manual | COMDTINST 1000.4C | 218 | Letter per chapter `A.`–`PP.` |
| CG Military Substance Abuse Manual | COMDTINST 1000.10B | 52 | Letter per chapter `A.` |
| CG Military Justice Manual | COMDTINST M5810.1H | ~262 | `Section A.` explicit |
| **Total** | | **~1,171** | |

All documents chunked at 600 tokens with 80–100 token overlap.

---

## System Architecture

```
User Query
    │
    ├─ Article detection    "Article 92" → filter to Art.92 chunks only
    ├─ Query expansion      "hazing" → adds "cruelty maltreatment Article 93"
    │                       "112a" → boosts exact BM25 token match
    │
    ▼ Stage 1 — Hybrid BM25 + Dense  (top-24 candidates)
    │
    │   Dense (FAISS)       Query → 384-dim vector → cosine similarity search
    │                       Strength: semantic paraphrase ("drug use" ↔ "wrongful use")
    │                       Weakness: exact legal codes ("112a", "87a") score ~0.05
    │
    │   BM25 (keyword)      Query tokens → exact term frequency scoring
    │                       Strength: legal codes, sub-article numbers, exact terms
    │                       Weakness: zero tolerance for paraphrase
    │
    │   RRF fusion          score = 1/(60 + dense_rank) + 1/(60 + bm25_rank)
    │
    ▼ Stage 2 — Cross-Encoder Rerank  (top-24 → top-8)
    │
    │   Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    │   Reads (query, passage) jointly — not as separate vectors
    │   Resolves adjacent-section errors and phrasing gaps that RRF cannot
    │   +8 HIT@1 over hybrid alone on the 200-question benchmark
    │
    ▼ Stage 3 — Answer
        Source View:   verbatim excerpts + section citations + page numbers
        AI Summary:    passages → Mistral 7B (temp=0.1) → structured plain-English summary
```

---

## Repository Structure

```
pio-rag/
│
├── src/
│   ├── chat_gui.py                       # GUI application (primary interface)
│   ├── chat.py                           # TUI chatbot (terminal fallback)
│   ├── query.py                          # Single-query with LLM synthesis
│   ├── query_wo_model.py                 # Single-query retrieval-only
│   │
│   ├── hybrid_retrieval.py               # BM25+dense RRF, article filter, query expansion
│   ├── reranker.py                       # Cross-encoder reranker (lazy-loads model)
│   ├── prompt_template.py                # LLM system prompt and demo questions
│   ├── eval_retrieval.py                 # Benchmark evaluation
│   │
│   ├── build_index.py                    # Build FAISS dense index
│   ├── build_bm25_index.py               # Build BM25 index (run after build_index.py)
│   │
│   ├── ingest_mcm.py                     # MCM 2024 — split by article number
│   ├── ingest_conduct_manual.py          # Conduct Manual — split by section 1.A.3.
│   ├── ingest_investigations_manual.py   # AIM — split by letter per chapter
│   ├── ingest_separations_manual.py      # Separations — split by letter, A.–PP.
│   ├── ingest_substance_abuse_manual.py  # SA Manual — split by letter per chapter
│   └── ingest_mjm.py                     # MJM — split by "Section A." headings
│
├── data/
│   ├── raw/                              # Source PDFs (not committed to repo)
│   ├── processed/                        # Per-document JSONL chunk files
│   └── index/
│       ├── pio_rag.faiss                 # FAISS dense index
│       ├── pio_rag_meta.json             # Chunk metadata (parallel to faiss)
│       └── pio_rag_bm25.pkl             # BM25 index
│
├── eval/
│   ├── benchmark_200.json                # 200-question full-corpus benchmark
│   ├── benchmark.json                    # 10-question MCM regression set
│   ├── demo_questions.json               # 10 curated demo questions with metadata
│   ├── demo_questions.txt                # Plain-text version for quick reference
│   └── results/                          # Saved eval run JSON files
│
├── assets/
│   └── CG_Legal_RAG_GUI_Image.png        # GUI background image
│
├── README.md
├── DEMO_GUIDE.md                         # Presentation day launch and talking points
└── PRESENTATION_EVAL.md                  # Benchmark methodology and results narrative
```

---

## Quick Start

### 1. Install dependencies

```bash
conda create -n pio-rag python=3.12 -y
conda activate pio-rag
pip install pypdf sentence-transformers faiss-cpu numpy rank_bm25 Pillow
# Optional — for LLM synthesis:
pip install llama-cpp-python
```

### 2. Ingest documents

```bash
python src/ingest_mcm.py                      --pdf data/raw/MCM_2024.pdf
python src/ingest_conduct_manual.py           --pdf data/raw/Conduct_Manual.pdf
python src/ingest_investigations_manual.py    --pdf data/raw/Investigations_Manual.pdf
python src/ingest_separations_manual.py       --pdf data/raw/Separations_Manual.pdf
python src/ingest_substance_abuse_manual.py   --pdf data/raw/Substance_Abuse_Manual.pdf
python src/ingest_mjm.py                      --pdf data/raw/MJM.pdf
```

### 3. Build indexes

```bash
python src/build_index.py        # FAISS dense index  (~30s)
python src/build_bm25_index.py   # BM25 index         (~10s)
```

Always run `build_bm25_index.py` after any ingestion change.

### 4. Launch the GUI

```bash
# Source View mode (no LLM required — recommended for demo):
python src/chat_gui.py --retriever hybrid --rerank

# With AI summaries (requires Mistral GGUF):
python src/chat_gui.py \
    --retriever hybrid \
    --rerank \
    --llm "C:\path\to\mistral-7b-instruct.Q4_K_M.gguf" \
    --mode llm
```

Place `CG_Legal_RAG_GUI_Image.png` in `assets/` for the background image.
See `DEMO_GUIDE.md` for full presentation instructions.

### 5. Single-query (command line)

```bash
# Retrieval-only:
python src/query_wo_model.py \
    "What are the elements of Article 92 failure to obey?" \
    --retriever hybrid --rerank

# With LLM synthesis:
python src/query.py \
    "What is the maximum punishment for larceny under Article 121?" \
    --retriever hybrid --rerank \
    --llm "C:\path\to\mistral.gguf"
```

### 6. Run evaluation

```bash
# Dense baseline:
python src/eval_retrieval.py \
    --benchmark eval/benchmark_200.json --ks 1 3 5 8 \
    --retriever dense \
    --out eval/results/run_dense.json

# Hybrid:
python src/eval_retrieval.py \
    --benchmark eval/benchmark_200.json --ks 1 3 5 8 \
    --retriever hybrid \
    --out eval/results/run_hybrid.json

# Hybrid + rerank (best confirmed result):
python src/eval_retrieval.py \
    --benchmark eval/benchmark_200.json --ks 1 3 5 8 \
    --retriever hybrid --rerank \
    --out eval/results/run_reranked.json

# Side-by-side comparison of all three:
python src/eval_retrieval.py \
    --compare eval/results/run_dense.json \
             eval/results/run_hybrid.json \
             eval/results/run_reranked.json
```

---

## Generation Settings

| Setting | Value | Rationale |
|---|---|---|
| Model | Mistral 7B Instruct Q4_K_M | Best quality/size for local CPU inference |
| Temperature | 0.1 | Near-deterministic — minimizes hallucination of legal facts |
| Max tokens | 1,500 | Sufficient for structured answer with all sections |
| Context window (n_ctx) | 8,192 | Accommodates system prompt + passages + response |
| Repeat penalty | 1.1 | Reduces verbatim echo of source text |

Do not raise temperature above 0.15 for legal text synthesis.

---

## Known Limitations

**1. General-domain embedding model.**
`all-MiniLM-L6-v2` was trained on general English, not military law. Exact legal
sub-article codes like `112a` or `87a` have near-zero dense similarity to their
descriptions (measured: cosine similarity ~0.05). BM25 token matching and the
query expansion table in `hybrid_retrieval.py` compensate for this.

**2. Terminology gaps require handwritten expansions.**
"Hazing" vs. "cruelty and maltreatment" has a measured cosine similarity of 0.14 —
dense search misses it entirely without the query expansion table.
20 such mappings are defined in `hybrid_retrieval.py`.

**3. One confirmed out-of-corpus question.**
q091 (VA disability offset of separation pay — 10 U.S.C. § 1174(h)) is not covered
in any ingested manual. The system returns the nearest available content.

**4. Cross-document questions are harder.**
The 9 cross-document benchmark questions require content from two manuals simultaneously.
The system retrieves mixed-source results and the LLM synthesizes across them, but
ranking both relevant sources in top-3 simultaneously is not always achieved.

**5. Reranker latency.**
Cross-encoder adds ~0.5s per query on a standard CPU. The full 200-question
reranked evaluation takes approximately 2–3 minutes on a laptop.

---

## Benchmark Methodology

`eval/benchmark_200.json` — 200 questions, 7 categories:

| Category | Count | Description |
|---|---|---|
| policy_lookup | 50 | What does policy X say |
| definition | 41 | What does term X mean |
| procedure | 35 | What are the steps for X |
| article_lookup | 33 | What does Article X cover |
| elements | 25 | What are the elements of Article X |
| cross_document | 9 | Requires content from two manuals |
| maximum_punishment | 7 | What is the max punishment for X |

A retrieval is scored correct (HIT@1) when the top-ranked chunk matches the gold
article or section label for the correct source document. Gold labels verified by
content inspection; `source_doc` field prevents cross-document false positives.

---

## Dependencies

```
pypdf >= 4.0
sentence-transformers >= 2.7      # includes CrossEncoder
faiss-cpu >= 1.8
numpy >= 1.26
rank_bm25 >= 0.2.2
Pillow >= 10.0                    # GUI background image
llama-cpp-python                  # optional — local LLM synthesis
```
