"""
pg_store.py
-----------
PostgreSQL + pgvector vector store cho Legal RAG.

Yêu cầu:
  - PostgreSQL >= 14 voi extension pgvector
  - Bien moi truong (hoac .env):
      POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

Schema bang legal_chunks:
  id            BIGSERIAL PRIMARY KEY
  chunk_id      TEXT UNIQUE           -- ma dinh danh ("Luat_DN_dieu4_part0")
  text          TEXT                  -- noi dung day du (breadcrumb + content)
  embedding     VECTOR(768)           -- vector embedding
  law_name      TEXT
  law_code      TEXT
  filename      TEXT
  part          TEXT
  chapter       TEXT
  chapter_title TEXT
  section       TEXT
  section_title TEXT
  article_id    TEXT                  -- "Dieu 4"
  article_num   INTEGER
  article_title TEXT
  breadcrumb    TEXT
  chunk_index   INTEGER
  total_chunks  INTEGER
  created_at    TIMESTAMPTZ DEFAULT NOW()
"""

from __future__ import annotations

import os
from typing import Optional

import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PgConnection

from src.chunking.legal_chunker import LegalChunk


# ──────────────────────────────────────────────────────────────────────────────
# Config (doc tu bien moi truong)
# ──────────────────────────────────────────────────────────────────────────────

def _get_dsn() -> str:
    """Xay dung PostgreSQL DSN tu bien moi truong."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB", "legal_rag")
    user = os.getenv("POSTGRES_USER", "postgres")
    pwd  = os.getenv("POSTGRES_PASSWORD", "postgres")
    return f"host={host} port={port} dbname={db} user={user} password={pwd}"


TABLE_NAME = "legal_chunks"
EMBEDDING_DIM = 768


# ──────────────────────────────────────────────────────────────────────────────
# SQL
# ──────────────────────────────────────────────────────────────────────────────

_SQL_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

_SQL_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id            BIGSERIAL PRIMARY KEY,
    chunk_id      TEXT UNIQUE NOT NULL,
    text          TEXT NOT NULL,
    embedding     VECTOR({EMBEDDING_DIM}),
    law_name      TEXT,
    law_code      TEXT,
    filename      TEXT,
    part          TEXT,
    chapter       TEXT,
    chapter_title TEXT,
    section       TEXT,
    section_title TEXT,
    article_id    TEXT,
    article_num   INTEGER,
    article_title TEXT,
    breadcrumb    TEXT,
    chunk_index   INTEGER DEFAULT 0,
    total_chunks  INTEGER DEFAULT 1,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
"""

_SQL_CREATE_INDEX = f"""
CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_embedding
ON {TABLE_NAME} USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
"""

_SQL_INSERT_CHUNK = f"""
INSERT INTO {TABLE_NAME} (
    chunk_id, text, embedding,
    law_name, law_code, filename,
    part, chapter, chapter_title,
    section, section_title,
    article_id, article_num, article_title,
    breadcrumb, chunk_index, total_chunks
) VALUES (
    %(chunk_id)s, %(text)s, %(embedding)s,
    %(law_name)s, %(law_code)s, %(filename)s,
    %(part)s, %(chapter)s, %(chapter_title)s,
    %(section)s, %(section_title)s,
    %(article_id)s, %(article_num)s, %(article_title)s,
    %(breadcrumb)s, %(chunk_index)s, %(total_chunks)s
)
ON CONFLICT (chunk_id) DO UPDATE SET
    text          = EXCLUDED.text,
    embedding     = EXCLUDED.embedding,
    law_name      = EXCLUDED.law_name,
    breadcrumb    = EXCLUDED.breadcrumb;
"""

_SQL_SIMILARITY_SEARCH = f"""
SELECT
    chunk_id, text, breadcrumb, law_name, law_code,
    article_id, article_title, chunk_index, total_chunks,
    1 - (embedding <=> %(query_vec)s::vector) AS score
FROM {TABLE_NAME}
{{where_clause}}
ORDER BY embedding <=> %(query_vec)s::vector
LIMIT %(k)s;
"""


# ──────────────────────────────────────────────────────────────────────────────
# PgVectorStore
# ──────────────────────────────────────────────────────────────────────────────

class PgVectorStore:
    """
    Vector store su dung PostgreSQL + pgvector.

    Sử dụng:
        store = PgVectorStore()
        store.setup()
        store.insert_chunks(chunks, embeddings)
        results = store.similarity_search(query_vec, k=5)
    """

    def __init__(self, dsn: str | None = None):
        self._dsn = dsn or _get_dsn()
        self._conn: Optional[PgConnection] = None

    def connect(self) -> None:
        """Mo ket noi toi PostgreSQL."""
        self._conn = psycopg2.connect(self._dsn)
        self._conn.autocommit = False
        print(f"  [PgStore] Ket noi thanh cong toi PostgreSQL.")

    def close(self) -> None:
        """Dong ket noi."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    # ---- Setup ----

    def setup(self) -> None:
        """
        Tao extension pgvector va bang legal_chunks neu chua co.
        An toan de chay nhieu lan (idempotent).
        """
        with self._conn.cursor() as cur:
            cur.execute(_SQL_CREATE_EXTENSION)
            cur.execute(_SQL_CREATE_TABLE)
        self._conn.commit()
        print(f"  [PgStore] Bang '{TABLE_NAME}' da san sang.")

    def create_index(self) -> None:
        """
        Tao IVFFlat index de tang toc do similarity search.
        Chi nen goi sau khi da insert du data (>= 1000 rows).
        """
        with self._conn.cursor() as cur:
            cur.execute(_SQL_CREATE_INDEX)
        self._conn.commit()
        print(f"  [PgStore] Index IVFFlat da duoc tao.")

    def drop_and_recreate(self) -> None:
        """Xoa sach va tao lai bang (dung khi re-ingest toan bo)."""
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
        self._conn.commit()
        self.setup()
        print(f"  [PgStore] Da xoa va tao lai bang '{TABLE_NAME}'.")

    # ---- Write ----

    def insert_chunks(
        self,
        chunks: list[LegalChunk],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> None:
        """
        Insert cac chunk va embedding vao PostgreSQL theo batch.

        Parameters
        ----------
        chunks : list[LegalChunk]
        embeddings : list[list[float]]
            Phai co cung do dai voi chunks.
        batch_size : int
            So rows moi lan commit (giam memory usage).
        """
        assert len(chunks) == len(embeddings), (
            f"So luong chunks ({len(chunks)}) va embeddings ({len(embeddings)}) khac nhau!"
        )

        total = len(chunks)
        inserted = 0

        with self._conn.cursor() as cur:
            for i in range(0, total, batch_size):
                batch_chunks = chunks[i: i + batch_size]
                batch_vecs   = embeddings[i: i + batch_size]

                rows = []
                for chunk, vec in zip(batch_chunks, batch_vecs):
                    row = chunk.to_metadata()
                    row["text"] = chunk.text
                    # pgvector nhan vecto dang string "[0.1, 0.2, ...]"
                    row["embedding"] = "[" + ",".join(str(v) for v in vec) + "]"
                    rows.append(row)

                psycopg2.extras.execute_batch(cur, _SQL_INSERT_CHUNK, rows)
                self._conn.commit()

                inserted += len(batch_chunks)
                print(f"  [PgStore] Da insert {inserted}/{total} chunks...", end="\r")

        print(f"\n  [PgStore] Hoan thanh insert {total} chunks.")

    # ---- Read ----

    def similarity_search(
        self,
        query_vec: list[float],
        k: int = 5,
        filter_law: list[str] | None = None,
    ) -> list[dict]:
        """
        Tim kiem cac chunk gan nhat voi query vector (cosine similarity).

        Parameters
        ----------
        query_vec : list[float]
            Vector cua cau hoi nguoi dung.
        k : int
            So ket qua tra ve.
        filter_law : list[str] | None
            Neu co, chi tim trong nhung luat cu the.
            Vi du: ["Luat Doanh Nghiep 2020", "Luat Dau Tu 2020"]

        Returns
        -------
        list[dict]
            Danh sach ket qua, moi item co: chunk_id, text, breadcrumb,
            law_name, article_id, score.
        """
        # Xây dựng mệnh đề WHERE nếu có filter
        where_clause = ""
        params: dict = {
            "query_vec": "[" + ",".join(str(v) for v in query_vec) + "]",
            "k": k,
        }

        if filter_law:
            placeholders = ", ".join(f"%s" for _ in filter_law)
            where_clause = f"WHERE law_name IN ({placeholders})"
            # psycopg2 can tuple for IN clause
            params["filter_laws"] = tuple(filter_law)
            where_clause = f"WHERE law_name = ANY(%(filter_laws)s)"

        sql = _SQL_SIMILARITY_SEARCH.format(where_clause=where_clause)

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [dict(r) for r in rows]

    def count(self) -> int:
        """Tra ve so luong chunks hien co trong bang."""
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
            return cur.fetchone()[0]

    def get_laws(self) -> list[str]:
        """Tra ve danh sach ten luat dang co trong store."""
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT law_name FROM {TABLE_NAME} ORDER BY law_name;")
            return [r[0] for r in cur.fetchall()]
