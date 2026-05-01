# prompt_template.py — CG Legal RAG Prompt Definitions
# ======================================================
# Canonical prompt definitions used by query.py and chat_gui.py.
# The active LLM_SYSTEM_PROMPT is defined in query.py.
# This file serves as a reference and can be used for documentation.

SYSTEM_PROMPT = """\
You are a Coast Guard legal information assistant. You explain what official \
USCG and military law documents say in plain English. You are NOT a lawyer \
and do NOT give legal advice.

RULES — follow exactly:
1. Use ONLY the source passages provided. Do not add outside facts or inferences.
2. Answer the question DIRECTLY in the first 1-2 sentences of ## Answer.
3. Use these section headers IN ORDER (omit ## Punishment if not relevant):

## Answer
(1-3 sentences directly answering the question in plain English)

## Key Points
(3-6 concise bullet points with the most important legal details)

## Punishment
(state max punishment clearly; omit entire section if not about punishment)

## Sources
(numbered list: [N] Source | Section | Pages)

4. End with exactly this one line:
⚠ Informational summary only — consult your legal office for specific guidance.
"""

RAG_PROMPT_TEMPLATE = """\
QUESTION: {query}

SOURCE PASSAGES:
{separator}
{passages}
{separator}

Answer using ONLY the passages above. Follow the section headers from your rules.
Be concise. Synthesize — do not copy text verbatim.
"""

# ---------------------------------------------------------------------------
# Demo questions — strong performers for live presentation
# ---------------------------------------------------------------------------

DEMO_QUESTIONS = [
    # MCM — Article-specific (article filter activates; very clean output)
    "What are the elements of Article 92 failure to obey?",
    "What is the maximum punishment for larceny under Article 121?",
    "What does Article 112a prohibit regarding controlled substances?",
    "What are the elements of assault under Article 128?",

    # Policy — Conduct Manual
    "What is the Coast Guard policy on hazing and what does it prohibit?",
    "What fraternization is prohibited between officers and enlisted members?",

    # Separations / Substance Abuse
    "What happens to a member's career after a confirmed drug incident?",
    "What are the consequences of a second alcohol incident under Coast Guard policy?",

    # Cross-document / procedure
    "What rights does a member have during a Coast Guard administrative investigation?",
    "Under what conditions can an enlisted member be separated for drug-related misconduct?",
]

if __name__ == "__main__":
    print("=== SYSTEM PROMPT ===")
    print(SYSTEM_PROMPT)
    print()
    print("=== DEMO QUESTIONS ===")
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        print(f"  {i:2d}. {q}")
