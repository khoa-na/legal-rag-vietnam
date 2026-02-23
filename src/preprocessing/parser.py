"""
parser.py
---------
Parse the hierarchical structure of Vietnamese legal documents.

Vietnamese law document structure:
  Law
  └── Phần / Part (optional, e.g. "Phần thứ nhất")
      └── Chương / Chapter (I, II, III…)
          └── Mục / Section (1, 2… — optional)
              └── Điều / Article (1, 2, 3…)
                  └── Khoản / Clause (1, 2, 3…)
                      └── Điểm / Point (a, b, c…)

This module:
  1. Detects law name and law code from the filename.
  2. Parses the cleaned text into a list of Article objects.
  3. Each Article carries its full structural context:
     law name, chapter, section, article ID and title.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .cleaner import clean_text


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Article:
    """
    Represents one Điều (article) — the basic chunking unit.

    Every Article carries enough structural metadata to reconstruct its
    breadcrumb and to populate vector-store metadata fields.
    """

    # Identity
    law_name:  str   # "Luật Doanh Nghiệp 2020"
    law_code:  str   # "59/2020/QH14"
    filename:  str   # "Luat_Doanh_Nghiep_2020.txt"

    # Structural position
    part:          Optional[str]  # "Phần thứ nhất"
    chapter:       Optional[str]  # "Chương I"
    chapter_title: Optional[str]  # "NHỮNG QUY ĐỊNH CHUNG"
    section:       Optional[str]  # "Mục 1"
    section_title: Optional[str]  # "CÔNG TY TNHH HAI THÀNH VIÊN TRỞ LÊN"

    # Article content
    article_id:    str  # "Điều 4"
    article_title: str  # "Giải thích từ ngữ"
    content:       str  # Full text of this article (all clauses)
    article_num:   int  # 4  (integer for sorting)

    def breadcrumb(self) -> str:
        """
        Build a human-readable context string.

        Example:
            "Luật Doanh Nghiệp 2020 | Chương I. NHỮNG QUY ĐỊNH CHUNG | Điều 4. Giải thích từ ngữ"
        """
        parts = [self.law_name]
        if self.part:
            parts.append(self.part)
        if self.chapter:
            chunk = self.chapter
            if self.chapter_title:
                chunk += f". {self.chapter_title}"
            parts.append(chunk)
        if self.section:
            chunk = self.section
            if self.section_title:
                chunk += f". {self.section_title}"
            parts.append(chunk)
        parts.append(f"{self.article_id}. {self.article_title}")
        return " | ".join(parts)

    def to_chunk_text(self) -> str:
        """Return the embeddable text: [breadcrumb] + content."""
        return f"[{self.breadcrumb()}]\n\n{self.content}"

    def to_dict(self) -> dict:
        """Flat dict for vector-store metadata."""
        return {
            "law_name":      self.law_name,
            "law_code":      self.law_code,
            "filename":      self.filename,
            "part":          self.part or "",
            "chapter":       self.chapter or "",
            "chapter_title": self.chapter_title or "",
            "section":       self.section or "",
            "section_title": self.section_title or "",
            "article_id":    self.article_id,
            "article_num":   self.article_num,
            "article_title": self.article_title,
            "breadcrumb":    self.breadcrumb(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ──────────────────────────────────────────────────────────────────────────────

# Part: "Phần thứ nhất.", "Phần thứ hai."
_RE_PART = re.compile(
    r'^(Phần\s+(?:thứ\s+)?(?:nhất|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|\d+))',
    re.MULTILINE | re.IGNORECASE,
)

# Chapter: "Chương I", "Chương II", "Chương 1"
_RE_CHAPTER = re.compile(
    r'^(Chương\s+(?:[IVXLCDM]+|\d+))[.\s]*$',
    re.MULTILINE,
)

# Chapter / section title: a long line starting with an uppercase letter,
# not a structural keyword, containing at least one ASCII uppercase letter.
_RE_CHAPTER_TITLE = re.compile(
    r'^(?!Điều|Chương|Mục|Khoản|Điểm|Phần|\d)'  # not a keyword or number
    r'(?=.*[A-Z])'                                # contains at least one uppercase ASCII letter
    r'.{5,}$'                                     # minimum 5 characters
)

# Section: "Mục 1", "Mục 2a"
_RE_SECTION = re.compile(
    r'^(Mục\s+\d+[a-z]?)[.\s]*$',
    re.MULTILINE,
)

# Article: "Điều 1.", "Điều 12a.", "Điều 198b."
_RE_ARTICLE = re.compile(
    r'^(Điều\s+(\d+[a-z]?))\.\s*(.*)$',
    re.MULTILINE,
)

# Law code in text: "59/2020/QH14"
_RE_LAW_CODE = re.compile(r'(\d+/\d{4}/QH\d+)')

# Filename stem -> friendly law name mapping
_FILENAME_TO_LAW_NAME: dict[str, str] = {
    "Luat_Canh_Tranh_2018":        "Luật Cạnh Tranh 2018",
    "Luat_Chung_Khoan_2019":       "Luật Chứng Khoán 2019",
    "Luat_Dau_Tu_2020":            "Luật Đầu Tư 2020",
    "Luat_Doanh_Nghiep_2020":      "Luật Doanh Nghiệp 2020",
    "Luat_Lao_Dong_2019":          "Bộ Luật Lao Động 2019",
    "Luat_Ngan_Sach_Nha_Nuoc_2025":"Luật Ngân Sách Nhà Nước 2025",
    "Luat_Quan_Ly_Thue_2019":      "Luật Quản Lý Thuế 2019",
    "Luat_So_Huu_Tri_Tue_2025":    "Luật Sở Hữu Trí Tuệ (sửa đổi 2025)",
    "Luat_Thuong_Mai_2025":        "Luật Thương Mại 2025",
}


# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

class LegalDocumentParser:
    """
    Parse a single Vietnamese law .txt file into a list of Article objects.

    Usage:
        parser   = LegalDocumentParser("docs/Luat_Doanh_Nghiep_2020.txt")
        articles = parser.parse()
        for art in articles:
            print(art.breadcrumb())
    """

    def __init__(self, file_path: str | Path):
        self.file_path     = Path(file_path)
        self.filename_stem = self.file_path.stem

        self.law_name = _FILENAME_TO_LAW_NAME.get(
            self.filename_stem,
            _normalize_filename_to_name(self.filename_stem),
        )
        self.law_code: str = ""

        # Mutable context state updated while walking through the document
        self._current_part:          Optional[str] = None
        self._current_chapter:       Optional[str] = None
        self._current_chapter_title: Optional[str] = None
        self._current_section:       Optional[str] = None
        self._current_section_title: Optional[str] = None

    def parse(self) -> list[Article]:
        """
        Read, clean, and parse the file.

        Returns
        -------
        list[Article]  in document order
        """
        raw   = self.file_path.read_text(encoding='utf-8')
        clean = clean_text(raw)

        # Extract law code from the first 500 characters
        code_match = _RE_LAW_CODE.search(clean[:500])
        if code_match:
            self.law_code = code_match.group(1)

        return self._parse_articles(clean)

    def _parse_articles(self, text: str) -> list[Article]:
        """
        Split the cleaned text into Article objects.

        For each article match:
          - content spans from this article's start to the next article's start
          - chapter / section context is derived from the preceding text
        """
        articles: list[Article] = []
        article_matches = list(_RE_ARTICLE.finditer(text))

        if not article_matches:
            return articles

        for idx, match in enumerate(article_matches):
            start = match.start()
            end   = (article_matches[idx + 1].start()
                     if idx + 1 < len(article_matches)
                     else len(text))

            article_id    = match.group(1).strip()
            article_num   = _parse_article_num(match.group(2).strip())
            article_title = match.group(3).strip()
            content       = text[start:end].strip()

            # Update chapter / section context from the text that precedes
            # this article
            self._update_context(text[:start])

            articles.append(Article(
                law_name=self.law_name,
                law_code=self.law_code,
                filename=self.file_path.name,
                part=self._current_part,
                chapter=self._current_chapter,
                chapter_title=self._current_chapter_title,
                section=self._current_section,
                section_title=self._current_section_title,
                article_id=article_id,
                article_title=article_title,
                content=content,
                article_num=article_num,
            ))

        return articles

    def _update_context(self, prefix_text: str) -> None:
        """
        Update Part / Chapter / Section state by scanning the prefix text.

        Finds the last match of each structural element so the context
        is always the most recent one before the current article.
        """
        # Part
        part_matches = list(_RE_PART.finditer(prefix_text))
        if part_matches:
            self._current_part = part_matches[-1].group(1).strip()

        # Chapter — only reset title/section when the chapter actually changes
        chapter_matches = list(_RE_CHAPTER.finditer(prefix_text))
        if chapter_matches:
            m           = chapter_matches[-1]
            new_chapter = m.group(1).strip()
            if new_chapter != self._current_chapter:
                self._current_chapter       = new_chapter
                self._current_chapter_title = None
                self._current_section       = None
                self._current_section_title = None

                # Chapter title: first non-empty line after the chapter header
                for line in prefix_text[m.end():].split('\n'):
                    stripped = line.strip()
                    if stripped:
                        if _RE_CHAPTER_TITLE.match(stripped):
                            self._current_chapter_title = stripped
                        break

        # Section — similarly, only reset when the section changes
        section_matches = list(_RE_SECTION.finditer(prefix_text))
        if section_matches:
            m           = section_matches[-1]
            new_section = m.group(1).strip()
            if new_section != self._current_section:
                self._current_section       = new_section
                self._current_section_title = None

                for line in prefix_text[m.end():].split('\n'):
                    stripped = line.strip()
                    if stripped:
                        if _RE_CHAPTER_TITLE.match(stripped):
                            self._current_section_title = stripped
                        break


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_filename_to_name(stem: str) -> str:
    """Fallback name derivation when the stem is not in the mapping dict."""
    return stem.replace('_', ' ').replace('Luat', 'Luật')


def _parse_article_num(s: str) -> int:
    """Extract the leading integer from an article number string."""
    match = re.match(r'(\d+)', s)
    return int(match.group(1)) if match else 0


# ──────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ──────────────────────────────────────────────────────────────────────────────

def parse_law_file(file_path: str | Path) -> list[Article]:
    """Parse a single law file and return its Article list."""
    return LegalDocumentParser(file_path).parse()


def parse_all_laws(docs_dir: str | Path) -> list[Article]:
    """
    Parse every .txt file in docs_dir and return all Articles.

    Parameters
    ----------
    docs_dir : path to the directory containing law .txt files

    Returns
    -------
    list[Article]  from all files, in alphabetical file order
    """
    docs_path    = Path(docs_dir)
    all_articles: list[Article] = []

    for f in sorted(docs_path.glob("*.txt")):
        articles = parse_law_file(f)
        all_articles.extend(articles)
        print(f"  [OK] {f.name}: {len(articles)} dieu")

    print(f"\nTong: {len(all_articles)} dieu tu {len(list(docs_path.glob('*.txt')))} file luat")
    return all_articles
