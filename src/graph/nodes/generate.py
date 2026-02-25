"""
generate.py
-----------
Generate Node — calls the LLM to produce a cited legal answer.

Also handles the "direct" route (no retrieval needed): the LLM answers
directly from its parametric knowledge for greetings / off-topic messages.
"""

from __future__ import annotations

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from src.graph.state import AgentState


# ──────────────────────────────────────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────────────────────────────────────

def _get_llm(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://ai.megallm.io/v1",
        api_key=os.environ.get("MEGALLM_API_KEY", ""),
        model="openai-gpt-oss-120b",
        temperature=temperature,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

_RAG_SYSTEM = """Bạn là chuyên gia tư vấn pháp luật Việt Nam.
Nhiệm vụ của bạn là trả lời câu hỏi pháp lý DỰA TRÊN CÁC ĐIỀU KHOẢN ĐƯỢC CUNG CẤP bên dưới.

Quy tắc bắt buộc:
1. Chỉ trả lời dựa trên văn bản pháp luật được cung cấp. Không bịa đặt.
2. Luôn ghi rõ nguồn trích dẫn: "Theo [Điều X], [Tên Luật]..."
3. Nếu thông tin không đủ để trả lời, hãy nói rõ: "Căn cứ thông tin hiện có, không tìm thấy quy định cụ thể về vấn đề này."
4. Câu trả lời rõ ràng, mạch lạc, dùng ngôn ngữ pháp lý chuẩn xác.
5. Không đưa ra tư vấn pháp lý cuối cùng — chỉ giải thích quy định.

Văn bản pháp luật liên quan:
{context}"""

_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _RAG_SYSTEM),
    ("human", "{question}"),
])

_DIRECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Bạn là trợ lý tư vấn pháp luật Việt Nam thân thiện. "
        "Hãy trả lời lịch sự và tự nhiên bằng tiếng Việt."
    )),
    ("human", "{question}"),
])


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _format_context(
    chunks: list[dict],
    sub_queries: list[str] | None = None,
    sub_contexts: list[list[dict]] | None = None,
) -> str:
    """Format retrieved chunks into a readable context block.

    If sub_queries and sub_contexts are provided (multi-hop), groups chunks
    by sub-query for clearer structure. Otherwise falls back to flat list.
    """
    if not chunks:
        return "(Không tìm thấy điều khoản liên quan.)"

    # Multi-hop: group by sub-query
    if sub_queries and sub_contexts and len(sub_queries) > 1:
        parts = []
        for sq_idx, (sq, sc) in enumerate(zip(sub_queries, sub_contexts), 1):
            parts.append(f"=== Sub-query {sq_idx}: {sq} ===")
            for i, chunk in enumerate(sc, 1):
                breadcrumb = chunk.get("breadcrumb", "")
                text       = chunk.get("text", "")
                parts.append(f"--- Nguồn {sq_idx}.{i}: {breadcrumb} ---\n{text}")
            parts.append("")  # blank line between groups
        return "\n\n".join(parts)

    # Simple: flat list (original behavior)
    parts = []
    for i, chunk in enumerate(chunks, 1):
        breadcrumb = chunk.get("breadcrumb", "")
        text       = chunk.get("text", "")
        parts.append(f"--- Nguồn {i}: {breadcrumb} ---\n{text}")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Node functions
# ──────────────────────────────────────────────────────────────────────────────

def generate_node(state: AgentState) -> dict:
    """
    LangGraph node: generate a cited answer from retrieved context.

    Called after the Retrieve node.
    """
    question     = state["question"]
    context      = state.get("context", [])
    sub_queries  = state.get("sub_queries", [])
    sub_contexts = state.get("sub_contexts", [])

    context_text = _format_context(context, sub_queries, sub_contexts)

    llm    = _get_llm()
    chain  = _RAG_PROMPT | llm
    result = chain.invoke({"question": question, "context": context_text})
    answer = result.content

    print(f"  [Generate] Answer generated ({len(answer)} chars).")
    return {
        "answer":   answer,
        "messages": [AIMessage(content=answer)],
    }


def direct_answer_node(state: AgentState) -> dict:
    """
    LangGraph node: answer directly without retrieval (for greetings, etc.).
    """
    question = state["question"]
    llm      = _get_llm(temperature=0.7)
    chain    = _DIRECT_PROMPT | llm
    result   = chain.invoke({"question": question})
    answer   = result.content

    print(f"  [Direct] Direct answer generated.")
    return {
        "answer":   answer,
        "messages": [AIMessage(content=answer)],
    }
