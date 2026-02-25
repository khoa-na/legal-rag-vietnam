"""
decompose.py
------------
Decompose Node — splits complex multi-hop questions into focused sub-queries.

When a user asks a comparative or multi-faceted legal question, this node
uses the LLM to break it into 2–4 focused sub-queries. Each sub-query is
then retrieved independently, giving better coverage across laws/topics.

Simple questions pass through unchanged (single-element sub_queries list).
"""

from __future__ import annotations

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.graph.state import AgentState


# ──────────────────────────────────────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://ai.megallm.io/v1",
        api_key=os.environ.get("MEGALLM_API_KEY", ""),
        model="openai-gpt-oss-120b",
        temperature=0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Decompose prompt
# ──────────────────────────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a query decomposition assistant for a Vietnamese legal Q&A system.\n\n"
        "Given a user's legal question, decide if it needs to be broken into simpler sub-queries.\n\n"
        "Rules:\n"
        "1. If the question is SIMPLE (asks about one specific topic, one law, one concept),\n"
        "   return the original question AS-IS on a single line.\n"
        "2. If the question is COMPLEX (compares across laws, asks about multiple concepts,\n"
        "   or requires information from different legal areas), break it into 2–4 focused\n"
        "   sub-queries. Each sub-query should be self-contained and specific.\n"
        "3. Keep every sub-query in Vietnamese.\n"
        "4. Output ONLY the sub-queries, one per line. No numbering, no bullets, no explanation.\n\n"
        "Examples:\n\n"
        "Input: \"Điều kiện thành lập công ty TNHH là gì?\"\n"
        "Output:\n"
        "Điều kiện thành lập công ty TNHH là gì?\n\n"
        "Input: \"So sánh quyền cổ đông theo Luật Doanh Nghiệp 2020 và Luật Chứng Khoán 2019\"\n"
        "Output:\n"
        "Quyền của cổ đông theo Luật Doanh Nghiệp 2020\n"
        "Quyền của cổ đông theo Luật Chứng Khoán 2019\n\n"
        "Input: \"Quy định về vốn điều lệ và thủ tục đăng ký doanh nghiệp theo Luật DN 2020\"\n"
        "Output:\n"
        "Quy định về vốn điều lệ theo Luật Doanh Nghiệp 2020\n"
        "Thủ tục đăng ký doanh nghiệp theo Luật Doanh Nghiệp 2020"
    )),
    ("human", "{question}"),
])


# ──────────────────────────────────────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────────────────────────────────────

def decompose_node(state: AgentState) -> dict:
    """
    LangGraph node: decompose a complex question into focused sub-queries.

    Simple questions pass through as a single-element list.
    Complex questions are split into 2–4 sub-queries for separate retrieval.

    Parameters
    ----------
    state : AgentState
        Must have "question" set.

    Returns
    -------
    dict  with "sub_queries" key — list of 1–4 sub-query strings
    """
    question = state["question"]

    llm   = _get_llm()
    chain = _DECOMPOSE_PROMPT | llm | StrOutputParser()
    raw   = chain.invoke({"question": question}).strip()

    # Parse: one sub-query per non-empty line
    sub_queries = [line.strip() for line in raw.split("\n") if line.strip()]

    # Safety: if LLM returns nothing useful, fall back to original question
    if not sub_queries:
        sub_queries = [question]

    # Cap at 4 sub-queries to avoid excessive retrieval
    sub_queries = sub_queries[:4]

    is_decomposed = len(sub_queries) > 1
    print(f"  [Decompose] {'Decomposed into ' + str(len(sub_queries)) + ' sub-queries' if is_decomposed else 'Simple question (pass-through)'}")
    for i, sq in enumerate(sub_queries, 1):
        print(f"    [{i}] {sq}")

    return {"sub_queries": sub_queries}
