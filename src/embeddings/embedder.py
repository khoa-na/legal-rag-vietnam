"""
embedder.py
-----------
HuggingFace embedding model wrapper for Vietnamese legal text.

Model  : dangvantuan/vietnamese-document-embedding
Source : HuggingFace (runs fully offline after first download)
Dim    : 768
Max seq: 8096 tokens

Usage:
    embedder = VietnameseEmbedder()
    vec  = embedder.embed_query("công ty cổ phần là gì?")
    vecs = embedder.embed_documents(["text 1", "text 2"])
    embedder.free_memory()   # release GPU/CPU VRAM after ingestion
"""

from __future__ import annotations

import gc
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME     = "dangvantuan/vietnamese-document-embedding"
MAX_SEQ_LENGTH = 8096   # tokens — fits almost all legal articles
EMBEDDING_DIM  = 768
BATCH_SIZE     = 8      # number of texts per forward pass (tune for VRAM)


# ──────────────────────────────────────────────────────────────────────────────
# Embedder
# ──────────────────────────────────────────────────────────────────────────────

class VietnameseEmbedder:
    """
    Wrapper around sentence-transformers for Vietnamese document embedding.

    Designed to be used once for ingestion (embed_documents) and then
    freed from memory. For query-time embedding, use embed_query.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_seq_length: int = MAX_SEQ_LENGTH,
        device: str | None = None,
        batch_size: int = BATCH_SIZE,
    ):
        self.model_name  = model_name
        self.batch_size  = batch_size

        # Auto-select device: CUDA > CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"  [Embedder] Loading model : {model_name}")
        print(f"  [Embedder] Device        : {device}")
        print(f"  [Embedder] Max seq length: {max_seq_length} tokens")

        self._model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self._model.max_seq_length = max_seq_length

    @property
    def dimension(self) -> int:
        """Output embedding dimension (768)."""
        return self._model.get_sentence_embedding_dimension()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string at retrieval time.

        Parameters
        ----------
        text : user query string

        Returns
        -------
        list[float] of length 768 (L2-normalized)
        """
        vec = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document chunks at ingestion time.

        Parameters
        ----------
        texts : list of chunk texts to embed

        Returns
        -------
        list of float vectors, each of length 768 (L2-normalized)
        """
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vecs]

    def free_memory(self) -> None:
        """
        Release the model from GPU / CPU memory after ingestion is complete.

        Call this immediately after embed_documents() to free VRAM before
        loading other models (e.g. the LLM).
        """
        del self._model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            print(f"  [Embedder] GPU VRAM freed. Remaining allocated: {allocated:.1f} MB")
        else:
            print(f"  [Embedder] CPU memory freed.")


@lru_cache(maxsize=1)
def get_embedder() -> VietnameseEmbedder:
    """
    Singleton embedder — loads the model only once per process lifetime.
    Use this in LangGraph retrieval nodes to avoid re-loading on every call.
    """
    return VietnameseEmbedder()
