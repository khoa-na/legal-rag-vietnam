"""
builder.py
----------
Assembles the LangGraph StateGraph for the Legal RAG Agent.

Graph topology:
                           ┌─────────────┐
              ──────────►  │   Router    │
                           └──────┬──────┘
                    "retrieve"    │    "direct"
                    ┌─────────────┤─────────────────┐
                    ▼                               ▼
             ┌──────────────┐              ┌───────────────┐
             │   Retrieve   │              │ Direct Answer │
             └──────┬───────┘              └───────────────┘
                    ▼
             ┌──────────────┐
             │   Generate   │
             └──────┬───────┘
                    ▼
             ┌──────────────┐
             │    Grader    │
             └──────┬───────┘
         pass/      │       fail + rewrite
         max_retry  │       ┌──────────────────────────┐
                    ▼       └──► (back to Retrieve)    │
                  END                                  │
                    ▲───────────────────────────────────┘
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from src.graph.state import AgentState
from src.graph.nodes.router import router_node
from src.graph.nodes.retrieve import retrieve_node
from src.graph.nodes.generate import generate_node, direct_answer_node
from src.graph.nodes.grader import grader_node


# ──────────────────────────────────────────────────────────────────────────────
# Conditional edge functions
# ──────────────────────────────────────────────────────────────────────────────

def _route_after_router(state: AgentState) -> str:
    """Branch after Router: 'retrieve' or 'direct'."""
    return state.get("route", "retrieve")


def _route_after_grader(state: AgentState) -> str:
    """Branch after Grader: loop back to 'retrieve' or end."""
    if state.get("needs_rewrite", False):
        return "retrieve"
    return END


# ──────────────────────────────────────────────────────────────────────────────
# Build and compile the graph
# ──────────────────────────────────────────────────────────────────────────────

def build_graph():
    """
    Build and compile the LangGraph StateGraph.

    Returns
    -------
    CompiledGraph  ready to call with .invoke() or .stream()
    """
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node("router",        router_node)
    g.add_node("retrieve",      retrieve_node)
    g.add_node("generate",      generate_node)
    g.add_node("grader",        grader_node)
    g.add_node("direct_answer", direct_answer_node)

    # Entry point
    g.set_entry_point("router")

    # Router → retrieve or direct_answer
    g.add_conditional_edges(
        "router",
        _route_after_router,
        {"retrieve": "retrieve", "direct": "direct_answer"},
    )

    # retrieve → generate
    g.add_edge("retrieve", "generate")

    # generate → grader
    g.add_edge("generate", "grader")

    # grader → retrieve (retry) or END
    g.add_conditional_edges(
        "grader",
        _route_after_grader,
        {"retrieve": "retrieve", END: END},
    )

    # direct_answer always ends
    g.add_edge("direct_answer", END)

    return g.compile()


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: run a single question
# ──────────────────────────────────────────────────────────────────────────────

def ask(question: str, filter_laws: list[str] | None = None) -> str:
    """
    Run the full LangGraph pipeline for a single question.

    Parameters
    ----------
    question    : user's question in Vietnamese
    filter_laws : optional list of law names to restrict retrieval

    Returns
    -------
    str  the final answer
    """
    graph = build_graph()

    initial_state: AgentState = {
        "messages":      [HumanMessage(content=question)],
        "question":      question,
        "context":       [],
        "answer":        "",
        "route":         "",
        "needs_rewrite": False,
        "rewrite_count": 0,
        "filter_laws":   filter_laws,
    }

    final_state = graph.invoke(initial_state)
    return final_state["answer"]
