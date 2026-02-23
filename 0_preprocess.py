"""
0_preprocess.py
---------------
Script chạy preprocessing toàn bộ bộ luật:
  1. Đọc tất cả file .txt trong docs/
  2. Làm sạch (clean) từng file
  3. Parse cấu trúc Chương/Điều/Khoản
  4. In báo cáo tổng quan và sample để kiểm tra chất lượng

Chạy:
  python 0_preprocess.py
  python 0_preprocess.py --sample 5        # In 5 article samples
  python 0_preprocess.py --law Luat_Doanh_Nghiep_2020  # Chỉ xử lý 1 luật
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Thêm root vào path để import src/
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import Article, parse_all_laws, parse_law_file

DOCS_DIR = ROOT / "docs"


# ──────────────────────────────────────────────────────────────────────────────
# Report helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(articles: list[Article]) -> None:
    """In thống kê tổng quan về corpus."""
    from collections import Counter

    law_counts = Counter(a.law_name for a in articles)
    total_tokens_est = sum(len(a.content.split()) for a in articles)

    print("\n" + "=" * 65)
    print("  PREPROCESSING SUMMARY")
    print("=" * 65)
    for law, count in sorted(law_counts.items()):
        print(f"  {law:<45} {count:>4} điều")
    print("-" * 65)
    print(f"  Tổng số điều:          {len(articles):>4}")
    print(f"  Ước tính từ (tổng):    {total_tokens_est:>8,}")
    print(f"  Trung bình từ/điều:    {total_tokens_est // max(len(articles), 1):>8,}")
    print("=" * 65)

    # Kiểm tra điều dài nhất
    longest = max(articles, key=lambda a: len(a.content))
    shortest = min(articles, key=lambda a: len(a.content))
    print(f"\n  Điều dài nhất:  {longest.article_id} ({longest.law_name})")
    print(f"                  {len(longest.content.split())} từ")
    print(f"  Điều ngắn nhất: {shortest.article_id} ({shortest.law_name})")
    print(f"                  {len(shortest.content.split())} từ")


def print_sample(articles: list[Article], n: int = 3) -> None:
    """In mẫu n article đầu tiên để kiểm tra chất lượng."""
    print(f"\n{'=' * 65}")
    print(f"  SAMPLE ({n} dieu dau tien)")
    print(f"{'=' * 65}")
    for art in articles[:n]:
        print(f"\nBREADCRUMB:\n   {art.breadcrumb()}")
        print(f"\nCONTENT (100 tu dau):\n")
        words = art.content.split()
        preview = " ".join(words[:100])
        if len(words) > 100:
            preview += " ..."
        print(f"   {preview}")
        print(f"\n   Metadata: {json.dumps(art.to_dict(), ensure_ascii=False, indent=2)}")
        print("-" * 65)


def check_quality(articles: list[Article]) -> None:
    """Kiểm tra chất lượng parsing — phát hiện điều thiếu tiêu đề."""
    print(f"\n{'=' * 65}")
    print("  QUALITY CHECK")
    print(f"{'=' * 65}")

    no_title = [a for a in articles if not a.article_title]
    no_chapter = [a for a in articles if not a.chapter]
    very_short = [a for a in articles if len(a.content.split()) < 10]

    print(f"  Điều thiếu tiêu đề:    {len(no_title)}")
    print(f"  Điều thiếu chương:     {len(no_chapter)}")
    print(f"  Điều quá ngắn (<10 từ): {len(very_short)}")

    if no_title:
        print("\n  [WARN] Dieu thieu tieu de:")
        for a in no_title[:5]:
            print(f"    - {a.article_id} | {a.law_name}")

    if very_short:
        print("\n  [WARN] Dieu qua ngan:")
        for a in very_short[:5]:
            print(f"    - {a.article_id} | {a.law_name}: '{a.content[:80]}'")

    if not no_title and not very_short:
        print("  [OK] Khong phat hien van de chat luong ro rang!")
    print("=" * 65)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline cho bộ văn bản luật Việt Nam"
    )
    parser.add_argument(
        "--law",
        type=str,
        default=None,
        help="Chỉ xử lý một file cụ thể, ví dụ: Luat_Doanh_Nghiep_2020",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=3,
        help="Số lượng article mẫu in ra (mặc định: 3)",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Bỏ qua phần in mẫu",
    )
    args = parser.parse_args()

    print(f"\nPreprocessing bo van ban luat Viet Nam")
    print(f"   Thư mục: {DOCS_DIR}\n")

    if args.law:
        # Xử lý một file cụ thể
        law_file = DOCS_DIR / f"{args.law}.txt"
        if not law_file.exists():
            print(f"[ERROR] Khong tim thay file: {law_file}")
            sys.exit(1)
        print(f"Dang xu ly: {law_file.name}")
        articles = parse_law_file(law_file)
        print(f"  [OK] {law_file.name}: {len(articles)} dieu")
    else:
        # Xử lý toàn bộ docs/
        articles = parse_all_laws(DOCS_DIR)

    # Báo cáo
    print_summary(articles)
    check_quality(articles)
    if not args.no_sample:
        print_sample(articles, n=args.sample)

    print(f"\n[DONE] Preprocessing hoan tat! {len(articles)} dieu luat san sang de ingest.")
    return articles


if __name__ == "__main__":
    main()
