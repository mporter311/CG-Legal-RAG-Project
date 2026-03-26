# prompt_template.py
# ===================
# Canonical RAG prompt template for the Coast Guard Legal RAG assistant.
# Sources: MCM 2024, CG Conduct Manual (COMDTINST M1600.2),
#          CG Separations Manual (COMDTINST 1000.4C).
#
# CHANGES FROM ORIGINAL (v2):
#   - Unified with query.py LLM_SYSTEM_PROMPT (was 4-section; now 5-section)
#   - Section order: summary / key elements / maximum punishment / citations / disclaimer
#   - Disclaimer wording locked to exact required text
#   - Source citation format updated to pull edition from metadata ("MCM" generic)
#   - Added RENDER EXAMPLE showing expected 5-section output structure
#
# This file is the single source of truth for the system prompt.
# query.py imports the constant LLM_SYSTEM_PROMPT from here (or keep in sync manually).

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a Coast Guard legal information assistant helping cadets and junior \
personnel understand U.S. military law. You are NOT a lawyer. You do NOT give \
legal advice. Your role is to explain, in plain language, what the official \
sources say.

STRICT RULES:
1. Use ONLY the provided source passages. Never add outside facts or inferences.
2. If passages are insufficient, say so explicitly and recommend the legal office.
3. Answer the QUESTION DIRECTLY in the first paragraph of your summary.
4. Explain what the article criminalizes or regulates.
5. When elements appear in the passages, list them as numbered bullets.
6. When maximum punishment appears, state it clearly and completely.
7. Write for a cadet audience: clear, precise, no unexplained legal jargon.
8. Every factual claim must trace to a citation using the source label from
   the passage header, e.g.:
   [MCM 2024 | Art. 92 | pp.337-339 | chunk: mcm2024_092_000]
   [CG Conduct Manual | Ch.2 > 2.A.4.c. | pp.108-108 | chunk: cgcm_ch02_0012]
9. Use EXACTLY these five section headers, in this order:

## Plain-language summary
## Key elements
## Maximum punishment
## Where this comes from (citations)
## Disclaimer

10. The Disclaimer section MUST read verbatim:
    "I am not a lawyer; this is an informational summary of official materials;
     consult your chain of command or legal office for advice."
"""

# ---------------------------------------------------------------------------
# RAG prompt template
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """\
{system_prompt}

============================================================
RETRIEVED SOURCE PASSAGES
============================================================
{passages}

============================================================
USER QUESTION
============================================================
{query}

Write your response using the five sections specified in your instructions.
Synthesize the passages — explain and summarise, do NOT just copy them verbatim.
"""

# ---------------------------------------------------------------------------
# Example rendered prompt (documentation / testing)
# ---------------------------------------------------------------------------

EXAMPLE_PASSAGES = """\
[Passage 1]
[MCM 2024 | Punitive Articles | Art. 92 — Failure to Obey Order or Regulation | pp.337-339 | chunk: mcm2024_092_000 | score: 0.921]

18. Article 92 (10 U.S.C. 892) — Failure to obey order or regulation

a. Text of statute. Any person subject to this chapter who—
(1) violates or fails to obey any lawful general order or regulation;
(2) having knowledge of any other lawful order issued by a member of the armed forces,
    which it is the member's duty to obey, fails to obey the order; or
(3) is derelict in the performance of duties;
shall be punished as a court-martial may direct.

b. Elements.
(1) Violation of or failure to obey a lawful general order or regulation.
  (a) That there was in effect a certain lawful general order or regulation;
  (b) That the accused had a duty to obey it; and
  (c) That the accused violated or failed to obey the order or regulation.
(2) Failure to obey other lawful order.
  (a) That a member of the armed forces issued a certain lawful order;
  (b) That the accused had knowledge of the order;
  (c) That the accused had a duty to obey the order; and
  (d) That the accused failed to obey the order.
(3) Dereliction of duty.
  (a) That the accused had certain duties;
  (b) That the accused knew or reasonably should have known of the duties; and
  (c) That the accused was (willfully) (through neglect or culpable inefficiency)
      derelict in the performance of those duties.

c. Maximum punishment.
(1) Violation of or failure to obey a lawful general order or regulation.
    Dishonorable discharge, forfeiture of all pay and allowances, and confinement for 2 years.
(2) Failure to obey other lawful order. Bad-conduct discharge, forfeiture of all pay and
    allowances, and confinement for 6 months.
(3) Dereliction of duty.
  (a) Through neglect or culpable inefficiency. Forfeiture of two-thirds pay per month for
      3 months and confinement for 3 months.
  (b) Willful. Bad-conduct discharge, forfeiture of all pay and allowances, and confinement
      for 6 months.
"""

EXAMPLE_RENDERED = RAG_PROMPT_TEMPLATE.format(
    system_prompt=SYSTEM_PROMPT,
    passages=EXAMPLE_PASSAGES,
    query="What does Article 92 cover and what are its elements?",
)

# ---------------------------------------------------------------------------
# Expected output structure (for documentation and evaluation rubric)
# ---------------------------------------------------------------------------

EXPECTED_OUTPUT_STRUCTURE = """
## Plain-language summary

Article 92 of the UCMJ (Failure to Obey Order or Regulation) covers three distinct
offenses: (1) violating a lawful general order or regulation, (2) failing to obey
any other lawful order, and (3) dereliction of duty. The article applies to all
persons subject to the UCMJ.

## Key elements

**Offense 1 — Violation of lawful general order or regulation:**
1. A lawful general order or regulation was in effect.
2. The accused had a duty to obey it.
3. The accused violated or failed to obey it.

**Offense 2 — Failure to obey other lawful order:**
1. A member of the armed forces issued a lawful order.
2. The accused had knowledge of the order.
3. The accused had a duty to obey it.
4. The accused failed to obey it.

**Offense 3 — Dereliction of duty:**
1. The accused had certain prescribed duties.
2. The accused knew or reasonably should have known of those duties.
3. The accused was willfully, or through neglect/culpable inefficiency,
   derelict in performing those duties.

## Maximum punishment

- Violation of lawful general order: Dishonorable discharge, forfeiture of all pay
  and allowances, and confinement for 2 years.
- Failure to obey other lawful order: Bad-conduct discharge, forfeiture of all pay
  and allowances, and confinement for 6 months.
- Dereliction (neglect/inefficiency): Forfeiture of 2/3 pay per month for 3 months
  and confinement for 3 months.
- Dereliction (willful): Bad-conduct discharge, forfeiture of all pay and allowances,
  and confinement for 6 months.

## Where this comes from (citations)

[1] [MCM 2024 | Punitive Articles | Art. 92 — Failure to Obey Order or Regulation |
    pp.337-339 | chunk: mcm2024_092_000 | score: 0.921]

## Disclaimer

I am not a lawyer; this is an informational summary of official materials;
consult your chain of command or legal office for advice.
"""

if __name__ == "__main__":
    print("=== SYSTEM PROMPT ===")
    print(SYSTEM_PROMPT)
    print()
    print("=== FULL RENDERED PROMPT (example, first 2500 chars) ===")
    print(EXAMPLE_RENDERED[:2500])
    if len(EXAMPLE_RENDERED) > 2500:
        print("[... truncated ...]")
    print()
    print("=== EXPECTED OUTPUT STRUCTURE ===")
    print(EXPECTED_OUTPUT_STRUCTURE)
