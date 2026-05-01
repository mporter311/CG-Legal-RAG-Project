# CG Legal RAG — Demo Guide

**Presentation Day Quick Reference**
**GUI Version — chat_gui.py**

---

## Pre-Demo Checklist

- [ ] `conda activate pio-rag`
- [ ] `cd` to repo directory
- [ ] Verify `data/index/pio_rag.faiss` exists → run `python src/build_index.py` if not
- [ ] Verify `data/index/pio_rag_bm25.pkl` exists → run `python src/build_bm25_index.py` if not
- [ ] Place background image at `assets/CG_Legal_RAG_GUI_Image.png` (create `assets/` folder if needed)
- [ ] If using LLM: verify GGUF file path, run one warm-up query before audience arrives
- [ ] Test launch command below — window should open within 10 seconds
- [ ] Run one demo question so the model is warm (first LLM call loads weights from disk)

---

## Launch Commands

### Source View Mode only (fastest — no LLM needed):
```bash
conda activate pio-rag
cd <repo-directory>
python src/chat_gui.py --retriever hybrid --rerank
```

### With AI summaries — recommended for full demo:
```bash
python src/chat_gui.py \
    --retriever hybrid \
    --rerank \
    --llm "C:\path\to\mistral-7b-instruct.Q4_K_M.gguf" \
    --mode llm
```
`--mode llm` opens directly in AI Summary Mode so you don't need to toggle on stage.

### Fastest startup (no reranking, no LLM):
```bash
python src/chat_gui.py --retriever hybrid
```

---

## GUI Controls

| Control | What it does |
|---|---|
| **Type + Enter** or **Send ▸** | Submit question, get answer |
| **Demo ▸** button | Cycles through 10 built-in demo questions one at a time |
| **[ Source View Mode ]** button | Toggles between Source View and AI Summary mode |
| **Clear** button | Clears the chat window, resets to welcome screen |
| **Click + drag** in chat area | Select and copy any text from the answer |

**Mode toggle button labels:**
- `[ Source View Mode ]` — currently showing verbatim retrieved passages
- `[  AI Summary Mode  ]` — currently using Mistral 7B to generate a structured summary

---

## Generation Settings (for reference)

| Setting | Value | Purpose |
|---|---|---|
| Temperature | 0.1 | Near-deterministic — prevents hallucination of legal facts |
| Max tokens | 1,500 | Enough for full structured answer with citations |
| Context window (n_ctx) | 8,192 | Accommodates system prompt + passages + response |
| Repeat penalty | 1.1 | Prevents echoing back source text verbatim |

---

## Recommended Demo Sequence (15 minutes)

### Opening (2 min) — Frame the problem
Say aloud: *"Suppose a cadet reports a hazing incident. Which manual do you check? Which article applies? Currently you'd search through multiple PDFs."*

Click **Demo ▸** or type:
> **"What is the Coast Guard policy on hazing and what conduct does it prohibit?"**

→ Shows: Conduct Manual content, citation with page number, instant retrieval

---

### Act 1 (3 min) — UCMJ article lookup

Type:
> **"What are the elements of Article 128 assault?"**

→ Shows: article filter activating, numbered legal elements, clean Source View display

Switch to AI Summary (click mode button), then type:
> **"What is the maximum punishment for larceny under Article 121?"**

→ Shows: Mistral 7B structured summary with `## Answer`, `## Key Points`, `## Punishment`, `## Sources`

---

### Act 2 (3 min) — Policy questions

Type:
> **"What fraternization is prohibited between officers and enlisted members?"**

→ Shows: Conduct Manual retrieval, real policy content, section citations

Type:
> **"What constitutes an alcohol incident under Coast Guard policy?"**

→ Shows: Substance Abuse Manual, precise regulatory definition

---

### Act 3 (3 min) — Cross-document questions

Type:
> **"What are the separation consequences for a second alcohol incident?"**

→ Shows: cross-encoder reranker fix in action (previously a retrieval miss, now rank #1)

Type:
> **"What rights does a member have during a Coast Guard administrative investigation?"**

→ Shows: AIM retrieval, cross-source content, Article 31(b) context

---

### Closing (4 min) — Architecture + Q&A

Show the evaluation slide. Talking points below.

---

## Talking Points

**On speed:**
> "Source View Mode responds in under 2 seconds. AI Summary with Mistral 7B takes about 15–30 seconds — it's running a 7-billion parameter model locally with no internet connection."

**On accuracy:**
> "We benchmarked 200 questions across all six documents. The hybrid retrieval pipeline correctly surfaces the right source at position 1 in 67% of cases. With the cross-encoder reranker, that rises to approximately 71%."

**On citations:**
> "Every answer includes the exact document, section number, and page range. The system only cites what it actually retrieved — it cannot invent a citation."

**On the AI summary:**
> "The LLM runs at temperature 0.1 — it's constrained to only summarize what's in the retrieved passages. It doesn't add outside knowledge or legal opinions."

**On limitations:**
> "This is a research prototype, not a replacement for legal counsel. The system says so explicitly at the end of every response. Questions requiring interpretation of a specific member's situation still need a JAG officer."

**On the ~30% it doesn't get right:**
> "Most remaining failures are where the correct passage is retrieved in the top 8 results but not ranked first. The cross-encoder reranker addresses the majority of those by re-scoring all candidates against the full query text before displaying results."

---

## 10 Optimized Demo Questions

These are the same questions loaded into the **Demo ▸** button. Each one has been validated for strong retrieval.

### Source View Mode (fast, verbatim — good for showing transparency)
| # | Question | Document |
|---|---|---|
| 1 | What are the elements of Article 92 failure to obey an order? | MCM 2024 |
| 3 | What does Article 112a prohibit regarding wrongful use of controlled substances? | MCM 2024 |
| 6 | What fraternization is prohibited between officers and enlisted members? | Conduct Manual |
| 8 | What are the separation consequences for a second alcohol incident? | Substance Abuse Manual |
| 9 | What rights does a member have during a Coast Guard administrative investigation? | AIM |

### AI Summary Mode (Mistral 7B — good for showing generation quality)
| # | Question | Document |
|---|---|---|
| 2 | What is the maximum punishment for larceny under Article 121? | MCM 2024 |
| 4 | What are the elements required to prove assault under Article 128? | MCM 2024 |
| 5 | What is the Coast Guard policy on hazing and what conduct does it prohibit? | Conduct Manual |
| 7 | What happens to a member's career after a confirmed drug incident finding? | Substance Abuse Manual |
| 10 | Under what conditions can an enlisted member be separated for drug-related misconduct? | Separations Manual |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Window doesn't open | Check `tkinter` is available: `python -c "import tkinter"` |
| Background image missing | Place `CG_Legal_RAG_GUI_Image.png` in `assets/` folder at repo root |
| "Index not found" error | Run `python src/build_index.py` then `python src/build_bm25_index.py` |
| LLM mode not available | Restart with `--llm "path/to/model.gguf"` — cannot be enabled after launch |
| LLM response cuts off | Normal for very long passages; raise `n_ctx` in `chat_gui.py` line ~230 if needed |
| First query is slow | Expected — model loads weights on first call (~10–30s). Run a warm-up before the demo. |
| Reranking feels slow on stage | Drop `--rerank` flag; hybrid-only is still strong for all 10 demo questions |
