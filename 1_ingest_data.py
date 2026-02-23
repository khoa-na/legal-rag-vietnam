"""
1_ingest_data.py
----------------
Full ingestion pipeline for the Vietnamese Legal RAG system:
  1. Parse all 9 law files into Article objects  (src/preprocessing)
  2. Chunk Articles into LegalChunk objects       (src/chunking)
  3. Embed chunks with vietnamese-document-embedding
  4. Store embeddings + metadata in ChromaDB      (src/embeddings/chroma_store)

Run:
  python 1_ingest_data.py                                  # all 9 laws
  python 1_ingest_data.py --law Luat_Doanh_Nghiep_2020    # single law
  python 1_ingest_data.py --reset                          # wipe & re-ingest
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import parse_all_laws, parse_law_file
from src.chunking.legal_chunker import chunk_all_articles
from src.embeddings.embedder import VietnameseEmbedder
from src.embeddings.chroma_store import ChromaVectorStore
from src.retrieval.bm25_retriever import BM25Retriever

DOCS_DIR = ROOT / "docs"


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_ingest(
    law_filter: str | None = None,
    reset: bool = False,
) -> None:
    t_start = time.time()

    print("\n" + "=" * 60)
    print("  LEGAL RAG — INGESTION PIPELINE")
    print("=" * 60)

    # ── Step 1: Parse ──────────────────────────────────────────────
    print("\n[1/4] Parsing legal documents...")
    if law_filter:
        law_file = DOCS_DIR / f"{law_filter}.txt"
        if not law_file.exists():
            print(f"[ERROR] File not found: {law_file}")
            sys.exit(1)
        articles = parse_law_file(law_file)
        print(f"  [OK] {law_file.name}: {len(articles)} articles")
    else:
        articles = parse_all_laws(DOCS_DIR)

    # Remove articles that have been abolished ("(được bãi bỏ)")
    before = len(articles)
    articles = [a for a in articles if "(được bãi bỏ)" not in a.content]
    skipped = before - len(articles)
    if skipped:
        print(f"  Skipped {skipped} abolished articles.")

    # ── Step 2: Chunk ──────────────────────────────────────────────
    print(f"\n[2/4] Chunking {len(articles)} articles...")
    chunks = chunk_all_articles(articles)
    print(f"  [OK] Total chunks: {len(chunks)}")

    # ── Step 3: Embed ──────────────────────────────────────────────
    print(f"\n[3/4] Embedding {len(chunks)} chunks...")
    print("  Model: dangvantuan/vietnamese-document-embedding")
    print("  This may take 5–30 minutes depending on hardware.")

    embedder = VietnameseEmbedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_documents(texts)

    # Release GPU / CPU VRAM immediately after embedding is done
    print("\n  Freeing embedding model from memory...")
    embedder.free_memory()

    print(f"  [OK] Embedding complete. Vector dimension: {len(embeddings[0])}")

    # ── Step 4: Store in ChromaDB ──────────────────────────────────
    print(f"\n[4/5] Saving to ChromaDB...")
    store = ChromaVectorStore()

    if reset:
        print("  [!] Reset mode: wiping existing collection...")
        store.reset()

    store.insert_chunks(chunks, embeddings)

    # Final report
    total_in_db = store.count()
    laws_in_db  = store.get_laws()

    print(f"\n{'=' * 60}")
    print(f"  RESULT:")
    print(f"  Total chunks in ChromaDB : {total_in_db}")
    print(f"  Laws indexed:")
    for law in laws_in_db:
        print(f"    - {law}")
    print(f"{'=' * 60}")

    # ── Step 5: Build BM25 index ───────────────────────────────────
    print(f"\n[5/5] Building BM25 sparse index...")
    BM25Retriever.load_or_build(force_rebuild=True)
    print(f"  [OK] BM25 index built and saved to disk.")

    elapsed = time.time() - t_start
    print(f"\n[DONE] Ingestion completed in {elapsed:.1f}s.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingestion pipeline: parse -> chunk -> embed -> ChromaDB"
    )
    parser.add_argument(
        "--law",
        type=str,
        default=None,
        help="Ingest a single law file, e.g.: Luat_Doanh_Nghiep_2020",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the ChromaDB collection and re-ingest from scratch",
    )
    args = parser.parse_args()

    run_ingest(law_filter=args.law, reset=args.reset)


if __name__ == "__main__":
    main()
