"""
bm25_retriever.py
-----------------
BM25 sparse retriever for Vietnamese legal document chunks.

Builds a BM25 index from all chunks stored in ChromaDB and saves it to disk
for fast reuse across runs. Rebuilt automatically if the index file is missing
or ChromaDB has been re-ingested.

BM25 excels at exact keyword matches (e.g. "doanh nghiệp nhà nước") that
dense vector search may miss when a chunk contains many diverse concepts.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_INDEX_PATH = Path(__file__).parent.parent.parent / "bm25_index.pkl"


# ──────────────────────────────────────────────────────────────────────────────
# BM25 Retriever
# ──────────────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    BM25 retriever that indexes all chunks from ChromaDB.

    Usage:
        retriever = BM25Retriever.load_or_build()
        results = retriever.search("doanh nghiệp nhà nước", k=10)
        # returns list of dicts matching ChromaDB's similarity_search format
    """

    def __init__(
        self,
        chunk_ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ):
        self.chunk_ids = chunk_ids
        self.texts     = texts
        self.metadatas = metadatas

        # Tokenize: simple whitespace split works well for Vietnamese
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 10,
        filter_law: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Return the top-k chunks by BM25 score.

        Parameters
        ----------
        query      : user query string
        k          : number of results
        filter_law : optional list of law names to restrict the search

        Returns
        -------
        list of dicts compatible with ChromaVectorStore.similarity_search output
        """
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # Build (idx, score) list, optionally filtering by law
        ranked = sorted(
            (
                (i, score)
                for i, score in enumerate(scores)
                if score > 0
                and (
                    filter_law is None
                    or self.metadatas[i].get("law_name") in filter_law
                )
            ),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        results = []
        for idx, score in ranked:
            meta = self.metadatas[idx]
            results.append({
                **meta,
                "text":  self.texts[idx],
                "score": float(score),       # BM25 raw score (not normalized)
                "distance": 0.0,             # placeholder for fusion
            })
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = DEFAULT_INDEX_PATH) -> None:
        """Persist the index to disk with pickle."""
        with open(path, "wb") as f:
            pickle.dump({
                "chunk_ids": self.chunk_ids,
                "texts":     self.texts,
                "metadatas": self.metadatas,
                "bm25":      self._bm25,
            }, f)
        print(f"  [BM25] Index saved → {path} ({len(self.texts)} docs)")

    @classmethod
    def load(cls, path: Path = DEFAULT_INDEX_PATH) -> "BM25Retriever":
        """Load a previously saved index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.chunk_ids = data["chunk_ids"]
        obj.texts     = data["texts"]
        obj.metadatas = data["metadatas"]
        obj._bm25     = data["bm25"]
        print(f"  [BM25] Index loaded from {path} ({len(obj.texts)} docs)")
        return obj

    @classmethod
    def load_or_build(
        cls,
        index_path: Path = DEFAULT_INDEX_PATH,
        force_rebuild: bool = False,
    ) -> "BM25Retriever":
        """
        Load the saved index if it exists, otherwise build from ChromaDB.

        Parameters
        ----------
        index_path    : path to the .pkl index file
        force_rebuild : if True, always rebuild and overwrite existing index
        """
        if not force_rebuild and index_path.exists():
            return cls.load(index_path)

        print("  [BM25] Building index from ChromaDB...")
        from src.embeddings.chroma_store import ChromaVectorStore
        store = ChromaVectorStore()

        # Fetch all documents from ChromaDB (metadata only, no vectors)
        result = store._collection.get(include=["documents", "metadatas"])

        chunk_ids = result["ids"]
        texts     = result["documents"]
        metadatas = result["metadatas"]

        obj = cls(chunk_ids, texts, metadatas)
        obj.save(index_path)
        return obj
