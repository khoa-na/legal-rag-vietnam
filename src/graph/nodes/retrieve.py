"""
retrieve.py
-----------
Retrieve Node — hybrid search combining BM25 (sparse) + ChromaDB (dense).

Results are fused using Reciprocal Rank Fusion (RRF), which combines the
ranked lists without requiring score normalisation.

Why hybrid?
  - Dense (ChromaDB): finds semantically similar chunks
  - Sparse (BM25)   : finds exact keyword matches (e.g. "doanh nghiệp nhà nước")
  - RRF             : each source contributes equally; no tuning required

NOTE: All heavy objects (ChromaDB store, embedding model, BM25 index) are
initialized lazily on the first call to avoid slow startup during import.
"""

from __future__ import annotations

from src.graph.state import AgentState

# Final number of chunks returned to the Generate node after fusion
TOP_K = 5

# Number of candidates fetched from each retriever before fusion
# Larger pool → better recall for RRF, at the cost of more context
CANDIDATE_K = 20

# RRF constant — k=60 is the standard default (Robertson et al.)
RRF_K = 60

# Lazy singletons
_store    = None
_embedder = None
_bm25     = None


# ──────────────────────────────────────────────────────────────────────────────
# Lazy singleton accessors
# ──────────────────────────────────────────────────────────────────────────────

def _get_store():
    global _store
    if _store is None:
        from src.embeddings.chroma_store import ChromaVectorStore
        _store = ChromaVectorStore()
    return _store


def _get_embedder():
    global _embedder
    if _embedder is None:
        from src.embeddings.embedder import get_embedder
        _embedder = get_embedder()
    return _embedder


def _get_bm25():
    global _bm25
    if _bm25 is None:
        from src.retrieval.bm25_retriever import BM25Retriever
        _bm25 = BM25Retriever.load_or_build()
    return _bm25


# ──────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ──────────────────────────────────────────────────────────────────────────────

def _rrf_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score = Σ  1 / (k + rank_i)
    where rank_i is the 1-indexed position in each result list.

    Returns results sorted by descending RRF score, preserving the full
    metadata dict from whichever list scored higher.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict]  = {}

    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk.get("chunk_id", str(rank))
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_map[cid]  = chunk

    for rank, chunk in enumerate(sparse_results, start=1):
        cid = chunk.get("chunk_id", str(rank))
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = chunk

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    fused = []
    for cid, rrf_score in ranked:
        entry = dict(chunk_map[cid])
        entry["rrf_score"] = rrf_score
        fused.append(entry)

    return fused


# ──────────────────────────────────────────────────────────────────────────────
# Node function
# ──────────────────────────────────────────────────────────────────────────────

def retrieve_node(state: AgentState) -> dict:
    """
    LangGraph node: hybrid retrieve (BM25 + dense) with RRF fusion.

    Parameters
    ----------
    state : AgentState
        Must have "question" set. Optionally "filter_laws" for scoped search.

    Returns
    -------
    dict  with updated "context" key — top-K fused chunks
    """
    question    = state["question"]
    filter_laws = state.get("filter_laws")

    # ── Dense retrieval (ChromaDB) ─────────────────────────────────────────
    query_vec     = _get_embedder().embed_query(question)
    dense_results = _get_store().similarity_search(
        query_vec,
        k=CANDIDATE_K,
        filter_law=filter_laws,
    )

    # ── Sparse retrieval (BM25) ────────────────────────────────────────────
    sparse_results = _get_bm25().search(
        question,
        k=CANDIDATE_K,
        filter_law=filter_laws,
    )

    # ── RRF Fusion ────────────────────────────────────────────────────────
    fused   = _rrf_fuse(dense_results, sparse_results)
    context = fused[:TOP_K]

    print(f"  [Retrieve] Hybrid search: dense={len(dense_results)}, "
          f"bm25={len(sparse_results)}, fused→top{TOP_K}")
    for i, c in enumerate(context):
        rrf   = c.get("rrf_score", 0)
        bc    = c.get("breadcrumb", "")[:75]
        print(f"    [{i+1}] rrf={rrf:.4f} | {bc}")

    return {"context": context}
