"""
state.py
--------
Defines the shared AgentState that flows through the LangGraph.

Every node reads from and writes to this state dict.
"""

from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state passed between LangGraph nodes.

    Fields
    ------
    messages     : Full conversation history (human + AI turns).
                   Uses add_messages reducer to append rather than overwrite.
    question     : The latest user question (extracted for convenience).
    context      : Retrieved document chunks (list of dicts from ChromaDB).
    answer       : Final generated answer string.
    route        : Decision from the Router node: "retrieve" | "direct".
    needs_rewrite: Flag set by Grader if the query needs to be rewritten.
    rewrite_count: Number of times the query has been rewritten (circuit-breaker).
    filter_laws  : Optional list of law names to restrict retrieval scope.
    """
    messages:      Annotated[list[BaseMessage], add_messages]
    question:      str
    context:       list[dict]
    answer:        str
    route:         str            # "retrieve" | "direct"
    needs_rewrite: bool
    rewrite_count: int
    filter_laws:   Optional[list[str]]
