"""
router.py
---------
Router Node — decides whether the question requires legal document retrieval
or can be answered directly (e.g. greetings, off-topic chit-chat).

Decision logic (LLM-based):
  - "retrieve" : legal / regulatory question → go to Retrieve node
  - "direct"   : greeting / off-topic → answer immediately without retrieval
"""

from __future__ import annotations

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.graph.state import AgentState


# ──────────────────────────────────────────────────────────────────────────────
# LLM (shared across nodes — MegaLLM OpenAI-compatible endpoint)
# ──────────────────────────────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://ai.megallm.io/v1",
        api_key=os.environ.get("MEGALLM_API_KEY", ""),
        model="openai-gpt-oss-120b",
        temperature=0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Router prompt
# ──────────────────────────────────────────────────────────────────────────────

_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a routing assistant for a Vietnamese legal Q&A system.\n"
        "Given the user's message, decide whether it requires searching through\n"
        "legal documents or can be answered directly.\n\n"
        "Respond with EXACTLY one word:\n"
        "  retrieve  — if the question is about laws, regulations, rights,\n"
        "              obligations, penalties, procedures, or any legal topic.\n"
        "  direct    — if it is a greeting, thank-you, off-topic question,\n"
        "              or small talk that does not require legal document search.\n\n"
        "Do NOT explain. Output only 'retrieve' or 'direct'."
    )),
    ("human", "{question}"),
])

_router_chain = _ROUTER_PROMPT | _get_llm() | StrOutputParser()


# ──────────────────────────────────────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    """
    LangGraph node: classify the question and set state["route"].

    Parameters
    ----------
    state : AgentState

    Returns
    -------
    dict  with updated "route" key
    """
    question = state["question"]
    raw      = _router_chain.invoke({"question": question}).strip().lower()
    route    = "retrieve" if "retrieve" in raw else "direct"

    print(f"  [Router] '{question[:60]}...' → {route}")
    return {"route": route}
