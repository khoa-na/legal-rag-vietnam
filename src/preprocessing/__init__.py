"""
src/preprocessing/__init__.py
Expose public API của preprocessing package.
"""
from .cleaner import clean_text, clean_file
from .parser import (
    Article,
    LegalDocumentParser,
    parse_law_file,
    parse_all_laws,
)

__all__ = [
    "clean_text",
    "clean_file",
    "Article",
    "LegalDocumentParser",
    "parse_law_file",
    "parse_all_laws",
]
