"""
2_test_graph.py
---------------
Interactive CLI to test the LangGraph Legal RAG agent.

Run:
    python 2_test_graph.py                        # interactive chat
    python 2_test_graph.py --question "..."       # single question
    python 2_test_graph.py --law "Luật DN 2020"   # filter by law

Requirements:
    MEGALLM_API_KEY environment variable must be set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.graph.builder import ask, build_graph


# ──────────────────────────────────────────────────────────────────────────────
# Quick validation
# ──────────────────────────────────────────────────────────────────────────────

def _check_api_key() -> None:
    key = os.environ.get("MEGALLM_API_KEY", "")
    if not key:
        print("[ERROR] MEGALLM_API_KEY is not set.")
        print("  Set it in your .env file or environment:")
        print("  MEGALLM_API_KEY=your_api_key_here")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Interactive chat loop
# ──────────────────────────────────────────────────────────────────────────────

def interactive_chat(filter_laws: list[str] | None = None) -> None:
    print("\n" + "=" * 60)
    print("  LEGAL RAG — Interactive Chat")
    print("  Model: openai-gpt-oss-120b (MegaLLM)")
    if filter_laws:
        print(f"  Scope: {', '.join(filter_laws)}")
    print("  Type 'exit' or Ctrl+C to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        print("\nThinking...\n")
        try:
            answer = ask(question, filter_laws=filter_laws)
            print(f"Agent:\n{answer}\n")
            print("-" * 60)
        except Exception as e:
            print(f"[ERROR] {e}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Batch test with predefined questions
# ──────────────────────────────────────────────────────────────────────────────

_TEST_QUESTIONS = [
    ("Xin chào bạn!", None),   # should go direct route
    ("Điều kiện thành lập công ty TNHH hai thành viên trở lên là gì?", ["Luật Doanh Nghiệp 2020"]),
    ("Người lao động được nghỉ phép năm bao nhiêu ngày?", ["Bộ Luật Lao Động 2019"]),
    ("Hành vi lạm dụng vị trí thống lĩnh thị trường bị xử lý như thế nào?", ["Luật Cạnh Tranh 2018"]),
    ("Thời hạn bảo hộ quyền tác giả đối với tác phẩm văn học là bao lâu?", ["Luật Sở Hữu Trí Tuệ (sửa đổi 2025)"]),
]

def run_tests() -> None:
    print("\n" + "=" * 60)
    print("  LEGAL RAG — Batch Test")
    print("=" * 60)

    for i, (q, laws) in enumerate(_TEST_QUESTIONS, 1):
        print(f"\n[Test {i}/{len(_TEST_QUESTIONS)}]")
        print(f"Q: {q}")
        if laws:
            print(f"   (filter: {laws})")
        try:
            answer = ask(q, filter_laws=laws)
            # Print first 400 chars of the answer
            preview = answer[:400] + ("..." if len(answer) > 400 else "")
            print(f"A: {preview}")
        except Exception as e:
            print(f"[ERROR] {e}")
        print("-" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _check_api_key()

    parser = argparse.ArgumentParser(
        description="Test the Legal RAG LangGraph agent."
    )
    parser.add_argument(
        "--question", "-q", type=str, default=None,
        help="Ask a single question and print the answer.",
    )
    parser.add_argument(
        "--law", "-l", type=str, default=None,
        help="Filter retrieval to a specific law name.",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run predefined batch test questions.",
    )
    args = parser.parse_args()

    filter_laws = [args.law] if args.law else None

    if args.test:
        run_tests()
    elif args.question:
        print(f"\nQ: {args.question}\n")
        answer = ask(args.question, filter_laws=filter_laws)
        print(f"A:\n{answer}")
    else:
        interactive_chat(filter_laws=filter_laws)


if __name__ == "__main__":
    main()
