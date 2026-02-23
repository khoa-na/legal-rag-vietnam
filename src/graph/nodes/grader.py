"""
grader.py
---------
Grader Node — evaluates the generated answer for quality.

Checks two things:
  1. Relevance  : Does the answer actually address the question?
  2. Grounding  : Is the answer supported by the retrieved context
                  (no hallucination)?

If either check fails and rewrite_count < MAX_REWRITES:
  → sets needs_rewrite=True so the graph loops back to Retrieve
    with a rewritten query.

If quality is acceptable (or max rewrites reached):
  → sets needs_rewrite=False so the graph terminates.
"""

from __future__ import annotations

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.graph.state import AgentState


MAX_REWRITES = 2   # circuit-breaker: stop looping after this many retries


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://ai.megallm.io/v1",
        api_key=os.environ.get("MEGALLM_API_KEY", ""),
        model="openai-gpt-oss-120b",
        temperature=0,
    )


# ── Grading prompt ─────────────────────────────────────────────────────────────

_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a grader evaluating a Vietnamese legal Q&A response.\n\n"
        "Given:\n"
        "  QUESTION  : the user's question\n"
        "  CONTEXT   : retrieved legal document chunks\n"
        "  ANSWER    : the generated answer\n\n"
        "Evaluate:\n"
        "  1. RELEVANCE  — does the answer address the question?\n"
        "  2. GROUNDING  — is every factual claim in the answer supported by\n"
        "                  the provided context (no hallucination)?\n\n"
        "Respond with exactly ONE word:\n"
        "  pass — answer is relevant and grounded\n"
        "  fail — answer is off-topic, incomplete, or contains hallucinations\n\n"
        "Do NOT explain. Only output 'pass' or 'fail'."
    )),
    ("human", (
        "QUESTION:\n{question}\n\n"
        "CONTEXT (first 2000 chars):\n{context_snippet}\n\n"
        "ANSWER:\n{answer}"
    )),
])

# ── Rewrite prompt ─────────────────────────────────────────────────────────────

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a query rewriter for a Vietnamese legal RAG system.\n"
        "The previous retrieval did not return useful results.\n"
        "Rewrite the user's question to improve document retrieval.\n"
        "Keep the legal intent intact. Output ONLY the rewritten question."
    )),
    ("human", "Original question: {question}"),
])


# ──────────────────────────────────────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────────────────────────────────────

def grader_node(state: AgentState) -> dict:
    """
    LangGraph node: grade the generated answer.

    If the answer fails grading and rewrite_count < MAX_REWRITES:
      - Rewrite the question for better retrieval.
      - Set needs_rewrite=True to loop back.

    If the answer passes or max rewrites exceeded:
      - Set needs_rewrite=False to end the graph.
    """
    question      = state["question"]
    answer        = state.get("answer", "")
    context       = state.get("context", [])
    rewrite_count = state.get("rewrite_count", 0)

    # Build a short context snippet for the prompt (avoid token overflow)
    context_snippet = "\n\n".join(
        c.get("text", "")[:500] for c in context[:3]
    )[:2000]

    llm   = _get_llm()
    chain = _GRADE_PROMPT | llm | StrOutputParser()
    grade = chain.invoke({
        "question":        question,
        "context_snippet": context_snippet,
        "answer":          answer,
    }).strip().lower()

    passed = "pass" in grade
    print(f"  [Grader] Grade: {'PASS ✓' if passed else 'FAIL ✗'}  "
          f"(rewrite_count={rewrite_count})")

    if passed or rewrite_count >= MAX_REWRITES:
        return {"needs_rewrite": False}

    # Rewrite the question for the next retrieval attempt
    rewrite_chain    = _REWRITE_PROMPT | llm | StrOutputParser()
    rewritten        = rewrite_chain.invoke({"question": question}).strip()
    print(f"  [Grader] Rewritten query: '{rewritten[:80]}'")

    return {
        "needs_rewrite": True,
        "question":      rewritten,
        "rewrite_count": rewrite_count + 1,
    }
