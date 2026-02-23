"""
legal_chunker.py
----------------
Split Article objects into LegalChunk objects ready for embedding.

Chunking strategy (3-tier):

  Tier 1 — Glossary articles ("Giai thich tu ngu" / many khoản):
    Each Khoản (clause) becomes its own chunk.
    Rationale: glossary articles pack 10-48 unrelated definitions into one
    text; a single vector would average across all of them and match nothing
    precisely. 1 definition = 1 focused vector.

  Tier 2 — Long articles (> MAX_WORDS and >= MIN_KHOAN_TO_GROUP khoản):
    Khoản are grouped into fixed-size windows (KHOAN_GROUP_SIZE) with a
    1-khoản overlap between consecutive chunks.

  Tier 3 — Normal articles (<= MAX_WORDS):
    1 article = 1 chunk (the sweet spot: median 225 words).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.preprocessing.parser import Article


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

# Tier-3 threshold: articles at or below this word count stay as 1 chunk.
# 600 words ≈ top ~90th percentile — covers the vast majority of articles.
MAX_WORDS_PER_CHUNK = 600

# Tier-1 detection: articles with this many khoản or more, OR whose title
# contains 'giai thich' / 'giải thích', are treated as glossary articles.
GLOSSARY_KHOAN_THRESHOLD = 10
_GLOSSARY_TITLE_KEYWORDS  = ('giải thích', 'giai thich', 'định nghĩa')

# Tier-2 grouping: number of khoản per chunk when splitting long articles.
KHOAN_GROUP_SIZE     = 4     # khoản per chunk
MIN_KHOAN_TO_GROUP   = 6     # only group-split if article has at least this many khoản

# Hard upper limit: even after grouping, a chunk must not exceed this.
ABS_MAX_WORDS = 6_000

# Leading clause pattern: "1. ", "2. ", "12. " at the start of a line
_RE_KHOAN = re.compile(r'^(\d+)\.\s', re.MULTILINE)


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LegalChunk:
    """
    The atomic unit stored in the vector database.

    In most cases one Article produces one LegalChunk.
    Very long Articles are split into multiple LegalChunks.
    """

    # Content
    chunk_id:     str   # unique ID: "<filename>_dieu<N>_part<P>"
    text:         str   # full embeddable text (breadcrumb header + content)

    # Metadata persisted alongside the vector
    law_name:      str
    law_code:      str
    filename:      str
    part:          str
    chapter:       str
    chapter_title: str
    section:       str
    section_title: str
    article_id:    str   # "Điều 4"
    article_num:   int
    article_title: str
    breadcrumb:    str
    chunk_index:   int   # 0 for unsplit articles, 0/1/2… for split
    total_chunks:  int   # total chunks produced from this article

    def to_metadata(self) -> dict:
        """Return a flat dict for storage as ChromaDB / pgvector metadata."""
        return {
            "chunk_id":     self.chunk_id,
            "law_name":     self.law_name,
            "law_code":     self.law_code,
            "filename":     self.filename,
            "part":         self.part,
            "chapter":      self.chapter,
            "chapter_title":self.chapter_title,
            "section":      self.section,
            "section_title":self.section_title,
            "article_id":   self.article_id,
            "article_num":  self.article_num,
            "article_title":self.article_title,
            "breadcrumb":   self.breadcrumb,
            "chunk_index":  self.chunk_index,
            "total_chunks": self.total_chunks,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def chunk_article(article: Article, max_words: int = MAX_WORDS_PER_CHUNK) -> list[LegalChunk]:
    """
    Convert one Article into one or more LegalChunks (3-tier strategy).

    Tier 1 — Glossary: split each Khoản individually.
    Tier 2 — Long article with many khoản: group K khoản per chunk.
    Tier 3 — Normal: 1 article = 1 chunk.
    """
    word_count     = len(article.content.split())
    khoan_matches  = list(_RE_KHOAN.finditer(article.content))
    n_khoan        = len(khoan_matches)
    title_lower    = article.article_title.lower()

    is_glossary = (
        any(kw in title_lower for kw in _GLOSSARY_TITLE_KEYWORDS)
        or n_khoan >= GLOSSARY_KHOAN_THRESHOLD
    )

    if is_glossary and n_khoan > 0:
        # Tier 1: 1 khoản = 1 chunk
        return _split_each_khoan(article, khoan_matches)

    if word_count > max_words and n_khoan >= MIN_KHOAN_TO_GROUP:
        # Tier 2: group KHOAN_GROUP_SIZE khoản per chunk
        return _split_khoan_groups(article, khoan_matches)

    if word_count > ABS_MAX_WORDS:
        # Safety net for very long articles with few khoản
        return _split_by_khoan(article, ABS_MAX_WORDS)

    # Tier 3: single chunk
    return [_make_chunk(article, article.content, chunk_index=0, total_chunks=1)]


def chunk_all_articles(
    articles: list[Article],
    max_words: int = MAX_WORDS_PER_CHUNK,
) -> list[LegalChunk]:
    """
    Chunk an entire list of Article objects.

    Parameters
    ----------
    articles  : output of parse_law_file / parse_all_laws
    max_words : word count threshold per chunk

    Returns
    -------
    list[LegalChunk]
    """
    all_chunks: list[LegalChunk] = []
    split_count = 0

    for article in articles:
        chunks = chunk_article(article, max_words)
        all_chunks.extend(chunks)
        if len(chunks) > 1:
            split_count += 1

    # Ensure all chunk_ids are globally unique.
    # Some consolidated laws (e.g. Luat_So_Huu_Tri_Tue_2025) include the same
    # article number in multiple parts/chapters, so we append _occN suffixes.
    dup_count = _deduplicate_chunk_ids(all_chunks)
    if dup_count:
        print(f"  [!] Resolved {dup_count} duplicate chunk_id(s) by appending occurrence suffixes.")

    print(f"  Total chunks: {len(all_chunks)} "
          f"(from {len(articles)} articles, {split_count} were split)")
    return all_chunks


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _deduplicate_chunk_ids(chunks: list[LegalChunk]) -> int:
    """
    Ensure every chunk_id in the list is unique.

    For duplicates the *first* occurrence keeps its original id.
    Subsequent occurrences get a suffix: ``_occ1``, ``_occ2``, etc.

    Returns the number of ids that were renamed.
    """
    seen: dict[str, int] = {}   # id -> occurrence count
    renamed = 0
    for chunk in chunks:
        base_id = chunk.chunk_id
        if base_id in seen:
            seen[base_id] += 1
            chunk.chunk_id = f"{base_id}_occ{seen[base_id]}"
            renamed += 1
        else:
            seen[base_id] = 0
    return renamed

def _make_chunk(
    article: Article,
    content: str,
    chunk_index: int,
    total_chunks: int,
) -> LegalChunk:
    """Build a LegalChunk from an Article and a content slice."""
    breadcrumb = article.breadcrumb()
    if total_chunks > 1:
        breadcrumb += f" (part {chunk_index + 1}/{total_chunks})"

    text = f"[{breadcrumb}]\n\n{content}"

    # Use the full article_id string (e.g. "Điều 12a") for the chunk_id
    # to avoid collisions between "Điều 12" and "Điều 12a" (both have article_num=12).
    safe_name       = article.filename.replace(".txt", "")
    safe_article_id = article.article_id.replace("Điều ", "dieu").replace(" ", "_")
    chunk_id        = f"{safe_name}_{safe_article_id}_part{chunk_index}"

    return LegalChunk(
        chunk_id=chunk_id,
        text=text,
        law_name=article.law_name,
        law_code=article.law_code,
        filename=article.filename,
        part=article.part or "",
        chapter=article.chapter or "",
        chapter_title=article.chapter_title or "",
        section=article.section or "",
        section_title=article.section_title or "",
        article_id=article.article_id,
        article_num=article.article_num,
        article_title=article.article_title,
        breadcrumb=article.breadcrumb(),
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )


def _split_by_khoan(article: Article, max_words: int) -> list[LegalChunk]:
    """
    Split an oversized Article at Khoản (clause) boundaries.

    Algorithm:
      1. Locate all clauses ("1. ", "2. "…) in the content.
      2. Group clauses greedily until max_words is reached.
      3. Add 1-clause overlap between consecutive groups for continuity.
    """
    khoan_matches = list(_RE_KHOAN.finditer(article.content))

    if not khoan_matches:
        # No recognisable clauses — brute-force split at midpoint
        words = article.content.split()
        mid   = len(words) // 2
        parts = [" ".join(words[:mid]), " ".join(words[mid:])]
        return [_make_chunk(article, p, i, len(parts)) for i, p in enumerate(parts)]

    # Extract each clause as a text block
    khoan_blocks: list[str] = []
    for i, m in enumerate(khoan_matches):
        start = m.start()
        end   = khoan_matches[i + 1].start() if i + 1 < len(khoan_matches) else len(article.content)
        khoan_blocks.append(article.content[start:end].strip())

    # Greedy grouping with 1-clause overlap
    groups: list[list[str]] = []
    current_group: list[str] = []
    current_words = 0

    for block in khoan_blocks:
        block_words = len(block.split())
        if current_words + block_words > max_words and current_group:
            groups.append(current_group)
            # Overlap: carry forward the last clause of the previous group
            current_group = [current_group[-1]]
            current_words = len(current_group[0].split())
        current_group.append(block)
        current_words += block_words

    if current_group:
        groups.append(current_group)

    # Prepend the article preamble (text before the first clause) to group 0
    preamble_end = khoan_matches[0].start()
    preamble     = article.content[:preamble_end].strip()
    total        = len(groups)
    chunks: list[LegalChunk] = []

    for i, group in enumerate(groups):
        content = "\n\n".join(group)
        if i == 0 and preamble:
            content = preamble + "\n\n" + content
        chunks.append(_make_chunk(article, content, chunk_index=i, total_chunks=total))

    return chunks


def _split_each_khoan(
    article: Article,
    khoan_matches: list,
) -> list[LegalChunk]:
    """
    Tier-1 split: each Khoản becomes its own chunk.

    Used for glossary articles ("Giải thích từ ngữ") where every clause is an
    independent definition. Keeping them separate produces focused, precise
    vectors instead of one averaged vector across dozens of concepts.

    Format of each chunk:
        [Breadcrumb]

        Điều X. Title — Khoản N
        N. <full text of this khoản>
    """
    # Preamble = text before the first khoản (usually just the article header)
    preamble_end = khoan_matches[0].start()
    preamble     = article.content[:preamble_end].strip()
    total        = len(khoan_matches)
    chunks: list[LegalChunk] = []

    for i, m in enumerate(khoan_matches):
        start = m.start()
        end   = khoan_matches[i + 1].start() if i + 1 < total else len(article.content)
        khoan_text = article.content[start:end].strip()

        # Include preamble only in the first chunk for context
        if i == 0 and preamble:
            content = preamble + "\n\n" + khoan_text
        else:
            content = khoan_text

        chunks.append(_make_chunk(article, content, chunk_index=i, total_chunks=total))

    return chunks


def _split_khoan_groups(
    article: Article,
    khoan_matches: list,
) -> list[LegalChunk]:
    """
    Tier-2 split: group KHOAN_GROUP_SIZE khoản per chunk with 1-khoản overlap.

    Used for long but non-glossary articles (>600 words, ≥6 khoản).
    Each chunk carries KHOAN_GROUP_SIZE consecutive clauses plus the last
    clause of the previous chunk for context continuity.
    """
    # Extract khoản text blocks
    blocks: list[str] = []
    n = len(khoan_matches)
    for i, m in enumerate(khoan_matches):
        start = m.start()
        end   = khoan_matches[i + 1].start() if i + 1 < n else len(article.content)
        blocks.append(article.content[start:end].strip())

    # Group into windows of KHOAN_GROUP_SIZE with 1-block overlap
    groups: list[list[str]] = []
    step = KHOAN_GROUP_SIZE          # advance by this many each time
    for start_idx in range(0, n, step):
        # Include previous block as overlap (except for the first group)
        group_start = max(0, start_idx - 1)
        group_end   = min(n, start_idx + KHOAN_GROUP_SIZE)
        groups.append(blocks[group_start:group_end])

    # Prepend preamble (article header) to the first group
    preamble_end = khoan_matches[0].start()
    preamble     = article.content[:preamble_end].strip()
    total        = len(groups)
    chunks: list[LegalChunk] = []

    for i, group in enumerate(groups):
        content = "\n\n".join(group)
        if i == 0 and preamble:
            content = preamble + "\n\n" + content
        chunks.append(_make_chunk(article, content, chunk_index=i, total_chunks=total))

    return chunks

