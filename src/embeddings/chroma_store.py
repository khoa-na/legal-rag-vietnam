"""
chroma_store.py
---------------
ChromaDB vector store for Legal RAG.

Installation:
    pip install chromadb langchain-chroma

Data is persisted locally to a directory (default: ./chroma_db).
No server setup required.

Usage:
    store = ChromaVectorStore()
    store.insert_chunks(chunks, embeddings)
    results = store.similarity_search(query_vec, k=5)

    # LangGraph integration:
    retriever = store.as_retriever(k=5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from src.chunking.legal_chunker import LegalChunk


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_PERSIST_DIR = Path(__file__).parent.parent.parent / "chroma_db"
COLLECTION_NAME = "legal_chunks"
EMBEDDING_DIM = 768


# ──────────────────────────────────────────────────────────────────────────────
# ChromaVectorStore
# ──────────────────────────────────────────────────────────────────────────────

class ChromaVectorStore:
    """
    Persistent ChromaDB vector store for legal document chunks.

    All data is saved to `persist_dir` on disk — no external server needed.
    Supports metadata filtering by law_name, chapter, article, etc.
    """

    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        print(f"  [ChromaDB] Persist dir : {self.persist_dir}")
        print(f"  [ChromaDB] Collection  : {self.collection_name}")
        print(f"  [ChromaDB] Existing docs: {self._collection.count()}")

    # ── Write ─────────────────────────────────────────────────────────────────

    def insert_chunks(
        self,
        chunks: list[LegalChunk],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> None:
        """
        Insert chunks with their embedding vectors into ChromaDB.

        Uses upsert — running again on the same data is safe (idempotent).

        Parameters
        ----------
        chunks     : list of LegalChunk objects
        embeddings : list of float vectors (must match length of chunks)
        batch_size : number of rows per upsert call to avoid memory spikes
        """
        assert len(chunks) == len(embeddings), (
            f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
        )

        total = len(chunks)
        inserted = 0

        for i in range(0, total, batch_size):
            batch_chunks = chunks[i: i + batch_size]
            batch_vecs   = embeddings[i: i + batch_size]

            self._collection.upsert(
                ids        = [c.chunk_id for c in batch_chunks],
                embeddings = batch_vecs,
                documents  = [c.text for c in batch_chunks],
                metadatas  = [c.to_metadata() for c in batch_chunks],
            )

            inserted += len(batch_chunks)
            print(f"  [ChromaDB] Upserted {inserted}/{total} chunks...", end="\r")

        print(f"\n  [ChromaDB] Done. Total in collection: {self._collection.count()}")

    def reset(self) -> None:
        """Delete and recreate the collection (wipes all data)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"  [ChromaDB] Collection reset.")

    # ── Read ──────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query_vec: list[float],
        k: int = 5,
        filter_law: list[str] | None = None,
    ) -> list[dict]:
        """
        Find the k nearest chunks to the query vector (cosine similarity).

        Parameters
        ----------
        query_vec  : embedding vector of the user query
        k          : number of results to return
        filter_law : optional list of law names to restrict search
                     e.g. ["Luật Doanh Nghiệp 2020", "Luật Đầu Tư 2020"]

        Returns
        -------
        list of dicts with keys: chunk_id, text, breadcrumb, law_name,
                                  article_id, distance, score
        """
        where: Optional[dict] = None
        if filter_law:
            if len(filter_law) == 1:
                where = {"law_name": {"$eq": filter_law[0]}}
            else:
                where = {"law_name": {"$in": filter_law}}

        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                **meta,
                "text": doc,
                "distance": dist,
                "score": 1.0 - dist,  # cosine distance -> similarity score
            })

        return output

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def get_laws(self) -> list[str]:
        """Return the distinct law names currently indexed."""
        # ChromaDB does not have a built-in distinct query,
        # so we fetch all metadatas and deduplicate in Python.
        # For large collections this is acceptable since metadata is small.
        results = self._collection.get(include=["metadatas"])
        seen: set[str] = set()
        for meta in results["metadatas"]:
            if meta.get("law_name"):
                seen.add(meta["law_name"])
        return sorted(seen)

    # ── LangChain / LangGraph integration ────────────────────────────────────

    def as_langchain_retriever(
        self,
        embedder,
        k: int = 5,
        filter_law: list[str] | None = None,
    ):
        """
        Return a LangChain-compatible VectorStoreRetriever for use in
        LangGraph nodes.

        Requires: pip install langchain-chroma

        Parameters
        ----------
        embedder   : VietnameseEmbedder instance (used to embed queries)
        k          : number of results per query
        filter_law : optional law name filter

        Example (in a LangGraph Retrieve node):
            retriever = store.as_langchain_retriever(embedder, k=5)
            docs = retriever.invoke("điều kiện thành lập công ty")
        """
        from langchain_chroma import Chroma
        from langchain_core.embeddings import Embeddings

        class _EmbedderAdapter(Embeddings):
            """Thin adapter from VietnameseEmbedder to LangChain Embeddings."""
            def embed_documents(self, texts):
                return embedder.embed_documents(texts)
            def embed_query(self, text):
                return embedder.embed_query(text)

        lc_store = Chroma(
            client=self._client,
            collection_name=self.collection_name,
            embedding_function=_EmbedderAdapter(),
        )

        search_kwargs: dict = {"k": k}
        if filter_law:
            if len(filter_law) == 1:
                search_kwargs["filter"] = {"law_name": filter_law[0]}
            else:
                search_kwargs["filter"] = {"law_name": {"$in": filter_law}}

        return lc_store.as_retriever(search_kwargs=search_kwargs)
