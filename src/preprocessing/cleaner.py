"""
cleaner.py
----------
Clean raw Vietnamese legal text files (.txt) before parsing.

Handles common OCR/formatting artifacts:
  - Annotation markers [1], [2], [3]... (amendment footnotes, not content)
  - Words split across blank lines: "Cổ\\n\\ntức" -> "Cổ tức"
  - Article header split from dot: "Điều 1\\n.\\nTitle" -> "Điều 1. Title"
  - Law number split by newline: "59\\n/2020/QH14" -> "59/2020/QH14"
  - Separator lines: ______, ------, ========
  - Boilerplate headers: QUỐC HỘI, CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM
  - Trailing/leading whitespace per line
  - Consecutive blank lines (collapsed to max 2)
"""

from __future__ import annotations

import re
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Compiled regex patterns
# ──────────────────────────────────────────────────────────────────────────────

# Amendment footnote markers: [1], [2a], [10]
_ANNOTATION = re.compile(r'\[\d+[a-z]?\]')

# Pure separator lines: ____, ----, ====
_SEPARATOR = re.compile(r'^[\s_\-=]+$', re.MULTILINE)

# Boilerplate header lines common to all law files
_BOILERPLATE_HEADER = re.compile(
    r'^(QUỐC HỘI|CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM|CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM'
    r'|Độc lập - Tự do - Hạnh phúc|Độc lập – Tự do – Hạnh phúc'
    r'|Bộ luật số:|Luật số:?)\s*$',
    re.MULTILINE,
)

# Law number split by newline: "59\n/2020" -> "59/2020"
_BROKEN_SLASH = re.compile(r'(\d)\r?\n+(/\d)', re.MULTILINE)

# Article dot split across lines: "Điều 1\n.\nTitle" -> "Điều 1. Title"
_BROKEN_ARTICLE_DOT = re.compile(
    r'(Điều\s+\d+[a-z]?|Khoản\s+\d+|Mục\s+\d+[a-z]?)\r?\n+\.\r?\n+',
    re.MULTILINE,
)

# Word split across a blank line (OCR artifact): "Cổ\n\ntức" -> "Cổ tức"
# Only applies when the continuation starts with a lowercase letter
_BROKEN_WORD = re.compile(
    r'([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ][a-zàáâãèéêìíòóôõùúýăđơư])\r?\n\r?\n([a-zàáâãèéêìíòóôõùúýăđơư])',
    re.MULTILINE,
)

# Collapse 3+ consecutive blank lines to 2
_MULTI_BLANK = re.compile(r'\n{3,}')

# Windows carriage return
_CR = re.compile(r'\r')


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    """
    Clean the raw contents of a single Vietnamese law file.

    Parameters
    ----------
    raw : str
        Raw UTF-8 text read from a .txt file.

    Returns
    -------
    str
        Normalised text ready for structural parsing.
    """
    text = raw

    # 1. Normalise line endings to \\n
    text = _CR.sub('', text)

    # 2. Remove amendment footnote markers [1], [2a], etc.
    text = _ANNOTATION.sub('', text)

    # 3. Rejoin broken law numbers: "59\n/2020/QH14" -> "59/2020/QH14"
    text = _BROKEN_SLASH.sub(r'\1\2', text)

    # 4. Rejoin broken article headers: "Điều 1\n.\nTitle" -> "Điều 1. Title"
    text = _BROKEN_ARTICLE_DOT.sub(r'\1. ', text)

    # 5. Rejoin OCR-split words: "Cổ\n\ntức" -> "Cổ tức"
    text = _BROKEN_WORD.sub(r'\1 \2', text)

    # 6. Remove boilerplate header lines
    text = _BOILERPLATE_HEADER.sub('', text)

    # 7. Remove pure separator lines
    text = _SEPARATOR.sub('', text)

    # 8. Strip trailing/leading whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # 9. Collapse consecutive blank lines
    text = _MULTI_BLANK.sub('\n\n', text)

    return text.strip()


def clean_file(file_path: str | Path) -> str:
    """
    Read and clean a single law .txt file.

    Parameters
    ----------
    file_path : str | Path

    Returns
    -------
    str  cleaned text content
    """
    path = Path(file_path)
    raw = path.read_text(encoding='utf-8')
    return clean_text(raw)
